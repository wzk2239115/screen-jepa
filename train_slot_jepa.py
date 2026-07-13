"""Slot Attention JEPA: word-level CLIP grounding with slot-attention-based
feature extraction instead of cell mean pooling.

Motivation: cell mean pooling produces flat word features (cos~0.97 in probe
distribution). Slot attention's competitive binding forces K slots to specialize
on different spatial regions (words / objects), producing sharper, more
discriminative word features. Also natural anti-collapse: slots compete,
cannot all collapse to the same point.

Pipeline:
  convnext -> feature map (196, D)
  slot attention -> K slots (K, D) + attention (196, K)
  word feat = word-mask-weighted slot attention mass @ slots
  photo feat = photo-cell-weighted slot attention @ slots
  InfoNCE: word <-> photo (cross-GPU gather)
  + JEPA masked latent prediction (MSE) + SIGReg
"""
import argparse
import io
import json
import random
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from render import DEFAULT_FONT
from backbones import build_encoder, TransformerBlock
from train import setup_ddp, lr_lambda
from model import SIGReg
from probe import ANTONYMS, SYNONYMS

from train_crossmodal_jepa import (
    W, H, build_composite, boxes_to_cell_masks,
    TarImageText, make_collate, region_mask, semantic_eval,
)


class SlotAttention(nn.Module):
    """Slot Attention (Locatello et al. 2020). Competitive attention: K slots
    compete to bind to spatial regions of the feature map. Returns slots and
    the final attention map (N, K)."""

    def __init__(self, dim, num_slots=16, num_iters=3, mlp_ratio=4):
        super().__init__()
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.dim = dim
        self.scale = dim ** -0.5

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.project_k = nn.Linear(dim, dim, bias=False)
        self.project_v = nn.Linear(dim, dim, bias=False)
        self.project_q = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim) * (dim ** -0.5))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, inputs, return_attn=False):
        # slot attention needs float32 for numerical stability (softmax, GRU)
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            inputs = inputs.float()
            B = inputs.size(0)
            K, D = self.num_slots, self.dim
            inp = self.norm_input(inputs)
            k = self.project_k(inp)
            v = self.project_v(inp)

            mu = self.slots_mu.expand(B, K, -1)
            sigma = self.slots_log_sigma.exp().expand(B, K, -1)
            slots = mu + sigma * torch.randn(B, K, D, device=inputs.device, dtype=torch.float32)

            last_attn = None
            for _ in range(self.num_iters):
                s = self.norm_slots(slots)
                q = self.project_q(s) * self.scale
                attn_logits = torch.einsum("bnd,bkd->bnk", k, q)
                attn = attn_logits.softmax(dim=-1)  # slots compete for each cell
                attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-6)  # normalize over cells

                updates = torch.einsum("bnk,bnd->bkd", attn, v)
                slots = self.gru(updates.reshape(-1, D), slots.reshape(-1, D)).reshape(B, K, D)
                slots = slots + self.mlp(self.norm_slots(slots))
                last_attn = attn

        if return_attn:
            return slots, last_attn
        return slots


class SlotJEPA(nn.Module):
    def __init__(self, arch="convnext", img_size=224, hidden=768, layers=12,
                 heads=12, patch=16, pred_depth=4, ema_tau=0.996,
                 num_slots=16, slot_iters=3):
        super().__init__()
        self.encoder = build_encoder(arch, img_size=img_size, dim=hidden,
                                     patch=patch, depth=layers, heads=heads)
        self.grid = getattr(self.encoder, "feature_grid", img_size // patch)
        self.feat_dim = getattr(self.encoder, "feature_dim", self.encoder.out_dim)
        g = self.grid
        self.text_cells = region_mask(g, list(range(g // 2)))
        self.photo_cells = region_mask(g, list(range(g // 2, g)))
        self.pos = nn.Parameter(torch.zeros(1, g * g, self.feat_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.feat_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.predictor = nn.Sequential(
            *[TransformerBlock(self.feat_dim, heads=max(1, self.feat_dim // 64))
              for _ in range(pred_depth)])
        self.pred_norm = nn.LayerNorm(self.feat_dim)
        self.target_encoder = self._copy(self.encoder)
        self.ema_tau = ema_tau
        self.sigreg = SIGReg()
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / 0.07)))
        self.slot_attn = SlotAttention(self.feat_dim, num_slots=num_slots, num_iters=slot_iters)

    @staticmethod
    def _copy(m):
        import copy
        c = copy.deepcopy(m)
        for p in c.parameters():
            p.requires_grad_(False)
        return c

    def update_ema(self):
        with torch.no_grad():
            for p, t in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                t.data.mul_(self.ema_tau).add_(p.data, alpha=1 - self.ema_tau)

    def encode(self, x):
        return self.encoder(x, return_map=True).mean(dim=1)

    def forward(self, composite, word_masks, word_valid, lam=0.0, w_mse=1.0, w_clip=1.0):
        with torch.no_grad():
            tgt = self.target_encoder(composite, return_map=True).detach()
        ctx = self.encoder(composite, return_map=True)
        B, N, D = ctx.shape

        # --- JEPA masked latent prediction ---
        def predict_region(mask_region):
            inp = ctx + self.pos
            m = mask_region.unsqueeze(0).expand(B, -1).unsqueeze(-1)
            inp = torch.where(m, self.mask_token.expand(B, N, D), inp)
            out = self.pred_norm(self.predictor(inp))
            return out[mask_region.unsqueeze(0).expand(B, -1)].reshape(B, -1, D), \
                   tgt[mask_region.unsqueeze(0).expand(B, -1)].reshape(B, -1, D)

        pred_t, tgt_t = predict_region(self.text_cells.to(ctx.device))
        pred_p, tgt_p = predict_region(self.photo_cells.to(ctx.device))
        mse_loss = F.mse_loss(pred_t, tgt_t) + F.mse_loss(pred_p, tgt_p)
        loss = w_mse * mse_loss
        stats = {}

        # --- word-level CLIP with slot attention ---
        if w_clip > 0 and word_masks is not None:
            slots, attn = self.slot_attn(ctx, return_attn=True)  # (B,K,D), (B,N,K)

            wm = word_masks.to(ctx.device).float()  # (B, max_w, N)
            wv = word_valid.to(ctx.device).float()  # (B, max_w)

            # word feature = word-mask-weighted slot attention @ slots
            word_slot_w = torch.einsum("bwn,bnk->bwk", wm, attn)  # (B, max_w, K)
            word_slot_w = word_slot_w / word_slot_w.sum(-1, keepdim=True).clamp(min=1e-6)
            word_feats = torch.einsum("bwk,bkd->bwd", word_slot_w, slots)  # (B, max_w, D)
            word_feats = F.normalize(word_feats, dim=-1)

            # photo global = photo-cell-weighted slot attention @ slots
            photo_idx = self.photo_cells.to(ctx.device)
            photo_attn = attn[:, photo_idx, :].mean(dim=1)  # (B, K)
            photo_attn = photo_attn / photo_attn.sum(-1, keepdim=True).clamp(min=1e-6)
            photo_global = torch.einsum("bk,bkd->bd", photo_attn, slots)  # (B, D)
            photo_global = F.normalize(photo_global, dim=-1)

            # gather photo_global across GPUs
            rank = 0
            photo_all = photo_global
            if dist.is_available() and dist.is_initialized():
                world = dist.get_world_size()
                if world > 1:
                    gathered = [torch.zeros_like(photo_global) for _ in range(world)]
                    dist.all_gather(gathered, photo_global.contiguous())
                    rank = dist.get_rank()
                    gathered[rank] = photo_global
                    photo_all = torch.cat(gathered, dim=0)

            logit_scale = self.logit_scale.exp().clamp(max=100.0)
            sim = torch.einsum("bwd,cd->bwc", word_feats, photo_all) * logit_scale
            Bm, max_w, Nb = sim.shape
            labels = torch.arange(Bm, device=ctx.device) + rank * Bm
            labels = labels.unsqueeze(1).expand(Bm, max_w)
            ce = F.cross_entropy(sim.reshape(Bm * max_w, Nb), labels.reshape(Bm * max_w), reduction="none")
            clip_loss = (ce.reshape(Bm, max_w) * wv).sum() / wv.sum().clamp(min=1)
            stats["clip"] = clip_loss.detach()
            loss = loss + w_clip * clip_loss

        if lam > 0:
            reg = self.sigreg(ctx.transpose(0, 1))
            loss = loss + lam * reg
            stats["reg"] = reg.detach()
        with torch.no_grad():
            stats["cos"] = 0.5 * (F.cosine_similarity(pred_t, tgt_t, dim=-1).mean()
                                  + F.cosine_similarity(pred_p, tgt_p, dim=-1).mean())
        return loss, stats


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tar_dir", required=True)
    p.add_argument("--num_tars", type=int, default=81)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--pred_depth", type=int, default=4)
    p.add_argument("--ema_tau", type=float, default=0.996)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--w_mse", type=float, default=0.3)
    p.add_argument("--w_clip", type=float, default=1.0)
    p.add_argument("--num_slots", type=int, default=16)
    p.add_argument("--slot_iters", type=int, default=3)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--bf16", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="./outputs/slot_jepa")
    return p.parse_args()


def main():
    args = build_args()
    amp = bool(args.bf16)
    rank, local_rank, world = setup_ddp()
    is_main = rank == 0
    device = f"cuda:{local_rank}"
    if is_main:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    grid = args.img_size // args.patch_size
    ds = TarImageText(args.tar_dir, args.num_tars, args.img_size,
                      grid=grid, patch_size=args.patch_size)
    collate = make_collate(grid)
    if world > 1:
        sampler = DistributedSampler(ds, shuffle=True, drop_last=True)
        train = DataLoader(ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                           drop_last=True, persistent_workers=args.workers > 0, pin_memory=True,
                           collate_fn=collate)
    else:
        train = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                           drop_last=True, persistent_workers=args.workers > 0, pin_memory=True,
                           collate_fn=collate)

    model = SlotJEPA("convnext", args.img_size, args.hidden, args.layers, args.heads,
                     args.patch_size, args.pred_depth, args.ema_tau,
                     args.num_slots, args.slot_iters).to(device)
    if world > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    base = model.module if isinstance(model, DDP) else model
    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[model] SlotJEPA trainable params(M)={n_params/1e6:.1f} "
              f"slots={args.num_slots} iters={args.slot_iters}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total_steps, args.warmup))

    step = 0
    for epoch in range(args.epochs):
        if world > 1:
            sampler.set_epoch(epoch)
        model.train()
        if is_main:
            bar = tqdm(train, desc=f"e{epoch}", dynamic_ncols=True, mininterval=2.0)
        else:
            bar = train
        for composite, word_masks, word_valid in bar:
            composite = composite.to(device, non_blocking=True)
            word_masks = word_masks.to(device, non_blocking=True)
            word_valid = word_valid.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp, dtype=torch.bfloat16):
                loss, stats = model(composite, word_masks, word_valid,
                                    lam=args.lam, w_mse=args.w_mse, w_clip=args.w_clip)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); sched.step()
            base.update_ema()
            if is_main and step % args.log_every == 0:
                clip_val = float(stats.get("clip", torch.tensor(0.0)))
                bar.set_postfix(loss=f"{loss.item():.4f}", cos=f"{float(stats['cos']):.3f}",
                                clip=f"{clip_val:.3f}", lr=f"{sched.get_last_lr()[0]:.1e}")
            step += 1
        if is_main:
            syn, ant, rnd = semantic_eval(base, device, amp, args.img_size, base.grid, DEFAULT_FONT)
            print(f"== epoch {epoch} semantic: syn={syn:.3f} ant={ant:.3f} random={rnd:.3f} "
                  f"(syn-rand={syn-rnd:+.3f}, ant-rand={ant-rnd:+.3f}) ==", flush=True)
            torch.save({"model": base.state_dict(),
                        "args": {**vars(args), "arch": "convnext", "objective": "slot_jepa"}},
                       out / f"epoch{epoch}.pt")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
