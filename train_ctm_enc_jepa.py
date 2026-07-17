"""CTM-Encoder JEPA: CTM iterative reasoning enhances the ENCODER feature map.

Key difference from train_ctm_jepa.py (CTM as predictor):
  - Before: CTM enhanced the PREDICTOR → only affected MSE loss → top-5 unchanged
  - Now: CTM enhances the ENCODER → affects CLIP loss (word-image alignment)
    → word features are extracted from a "thoughtfully processed" feature map

Pipeline:
  合成图 → ConvNeXt → initial feature map (B,N,D)
                        ↓
          CTM iterative enhancement (K thoughts × T ticks)
            thoughts attend to feature map → update → broadcast back
                        ↓
          enhanced feature map (B,N,D)
                        ↓
    ┌──── cell pooling → 词特征 + 图片特征 → CLIP InfoNCE
    └──── mask + Transformer predictor → JEPA MSE (target from EMA encoder)
"""
import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from render import DEFAULT_FONT
from backbones import build_encoder, TransformerBlock
from train import setup_ddp, lr_lambda
from model import SIGReg

from train_crossmodal_jepa import (
    build_composite, boxes_to_cell_masks,
    TarImageText, make_collate, region_mask, semantic_eval,
)
from train_ctm_jepa import SuperLinear


class CTMEnhancer(nn.Module):
    """CTM iterative enhancement for encoder feature maps.

    K thought tokens iterate T ticks, attending to the N-cell feature map.
    Final broadcast: cells query thoughts → enhanced feature map.
    Per-tick cost O(K*N*D) enables T=50+ ticks efficiently.
    """

    def __init__(self, dim, num_tokens, num_iters=50, memory_length=4,
                 num_thoughts=8, num_heads=4, bptt_window=15):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.num_thoughts = num_thoughts
        self.num_iters = num_iters
        self.memory_length = memory_length
        self.num_heads = num_heads
        self.bptt_window = bptt_window
        self.use_checkpoint = False

        self.thought_init = nn.Parameter(torch.zeros(1, num_thoughts, dim))
        nn.init.normal_(self.thought_init, std=0.02)

        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.attn_out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)

        self.synapse = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.LayerNorm(dim))

        self.nlm = SuperLinear(memory_length, dim)
        self.nlm_norm = nn.LayerNorm(dim)

        # broadcast: cells query thoughts
        self.out_q_proj = nn.Linear(dim, dim)
        self.out_kv_proj = nn.Linear(dim, dim * 2)
        self.out_attn_proj = nn.Linear(dim, dim)
        self.out_norm = nn.LayerNorm(dim)

    def _tick(self, thoughts, kv_input, trace):
        q = self.q_proj(thoughts)
        kv = self.kv_proj(kv_input)
        k, v = kv.chunk(2, dim=-1)
        B, K, D = q.shape
        N = k.size(1)
        hd = D // self.num_heads
        qh = q.reshape(B, K, self.num_heads, hd).transpose(1, 2)
        kh = k.reshape(B, N, self.num_heads, hd).transpose(1, 2)
        vh = v.reshape(B, N, self.num_heads, hd).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(qh, kh, vh)
        attn_out = attn_out.transpose(1, 2).reshape(B, K, D)
        attn_out = self.attn_out_proj(attn_out)
        thoughts = self.norm1(thoughts + attn_out)
        thoughts = self.synapse(torch.cat([thoughts, attn_out], dim=-1))
        trace = torch.cat([trace[..., 1:], thoughts.unsqueeze(-1)], dim=-1)
        nlm_out = self.nlm(trace)
        thoughts = thoughts + self.nlm_norm(nlm_out)
        return thoughts, trace

    def forward(self, ctx):
        """ctx: (B, N, D) → enhanced (B, N, D)"""
        B, N, D = ctx.shape
        K, M = self.num_thoughts, self.memory_length

        thoughts = self.thought_init.expand(B, K, D)
        trace = thoughts.unsqueeze(-1).expand(B, K, D, M).contiguous()

        for t in range(self.num_iters):
            if self.training and t > 0 and t % self.bptt_window == 0:
                thoughts = thoughts.detach()
                trace = trace.detach()
            if self.training and self.use_checkpoint:
                thoughts, trace = checkpoint(self._tick, thoughts, ctx, trace, use_reentrant=False)
            else:
                thoughts, trace = self._tick(thoughts, ctx, trace)

        # broadcast back to all cells
        q = self.out_q_proj(ctx)
        kv = self.out_kv_proj(thoughts)
        k, v = kv.chunk(2, dim=-1)
        hd = D // self.num_heads
        qh = q.reshape(B, N, self.num_heads, hd).transpose(1, 2)
        kh = k.reshape(B, K, self.num_heads, hd).transpose(1, 2)
        vh = v.reshape(B, K, self.num_heads, hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(qh, kh, vh)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_norm(ctx + self.out_attn_proj(out))


class CTMEncoderJepa(nn.Module):
    """Cross-modal JEPA with CTM-enhanced encoder."""

    def __init__(self, arch="convnext", img_size=224, hidden=768, layers=12,
                 heads=12, patch=16, pred_depth=4, ema_tau=0.996,
                 ctm_iters=50, ctm_memory=4, ctm_thoughts=8, bptt_window=15,
                 freeze_enhancer=False):
        super().__init__()
        self.freeze_enhancer = freeze_enhancer
        # backbone (standard ConvNeXt)
        self.backbone = build_encoder(arch, img_size=img_size, dim=hidden,
                                      patch=patch, depth=layers, heads=heads)
        self.grid = getattr(self.backbone, "feature_grid", img_size // patch)
        self.feat_dim = getattr(self.backbone, "feature_dim", self.backbone.out_dim)
        g = self.grid

        # CTM enhancer (iterative reasoning on feature map)
        self.enhancer = CTMEnhancer(
            self.feat_dim, g * g, num_iters=ctm_iters, memory_length=ctm_memory,
            num_thoughts=ctm_thoughts, num_heads=max(1, self.feat_dim // 64),
            bptt_window=bptt_window)

        self.text_cells = region_mask(g, list(range(g // 2)))
        self.photo_cells = region_mask(g, list(range(g // 2, g)))
        self.pos = nn.Parameter(torch.zeros(1, g * g, self.feat_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.feat_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        # simple Transformer predictor (4 layers, no CTM here)
        self.predictor = nn.Sequential(
            *[TransformerBlock(self.feat_dim, heads=max(1, self.feat_dim // 64))
              for _ in range(pred_depth)])
        self.pred_norm = nn.LayerNorm(self.feat_dim)

        # EMA target (backbone + enhancer)
        self.target_backbone = self._copy(self.backbone)
        self.target_enhancer = self._copy(self.enhancer)
        if self.freeze_enhancer:
            for p in self.enhancer.parameters():
                p.requires_grad_(False)
        self.ema_tau = ema_tau
        self.sigreg = SIGReg()
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / 0.07)))

    @staticmethod
    def _copy(m):
        import copy
        c = copy.deepcopy(m)
        for p in c.parameters():
            p.requires_grad_(False)
        return c

    def update_ema(self, epoch=0, total_epochs=1):
        # EMA tau annealing: 0.996 → 1.000 (I-JEPA style, target gets more stable)
        import math
        progress = min(epoch / max(total_epochs, 1), 1.0)
        tau = self.ema_tau + (1.0 - self.ema_tau) * (1 - math.cos(math.pi * progress)) / 2
        with torch.no_grad():
            for p, t in zip(self.backbone.parameters(), self.target_backbone.parameters()):
                t.data.mul_(tau).add_(p.data, alpha=1 - tau)
            if not self.freeze_enhancer:
                for p, t in zip(self.enhancer.parameters(), self.target_enhancer.parameters()):
                    t.data.mul_(tau).add_(p.data, alpha=1 - tau)

    def _encode(self, x, target=False):
        """Encode image → enhanced feature map (B, N, D)."""
        if target:
            ctx = self.target_backbone(x, return_map=True).detach()
            ctx = self.target_enhancer(ctx).detach()
        else:
            ctx = self.backbone(x, return_map=True)
            ctx = self.enhancer(ctx)
        return ctx

    def encode(self, x):
        return self._encode(x).mean(dim=1)

    def encoder(self, x, return_map=False):
        """Compatibility for semantic_eval / probe code expecting model.encoder."""
        ctx = self._encode(x, target=False)
        if return_map:
            return ctx
        return ctx.mean(dim=1)

    def forward(self, composite, word_masks=None, word_valid=None, lam=0.0, w_mse=1.0, w_clip=1.0,
                loss_type="siglip", align_mode="sentence", split_grad=True):
        ctx_raw = self.backbone(composite, return_map=True)  # (B, N, D)
        ctx_enh = self.enhancer(ctx_raw)  # (B, N, D)

        if split_grad:
            # JEPA on RAW features → gradient to backbone only (NOT enhancer)
            ctx_jepa = ctx_raw
            with torch.no_grad():
                tgt_jepa = self.target_backbone(composite, return_map=True).detach()
        else:
            # UNIFIED: JEPA on ENHANCED features → gradient to backbone + enhancer
            ctx_jepa = ctx_enh
            with torch.no_grad():
                tgt_raw = self.target_backbone(composite, return_map=True).detach()
                tgt_jepa = self.target_enhancer(tgt_raw).detach()

        B, N, D = ctx_jepa.shape
        dev = ctx_jepa.device

        def predict_region(mask_region):
            inp = ctx_jepa + self.pos
            m = mask_region.to(dev).unsqueeze(0).expand(B, -1).unsqueeze(-1)
            inp = torch.where(m, self.mask_token.expand(B, N, D), inp)
            out = self.pred_norm(self.predictor(inp))
            mr = mask_region.to(dev).unsqueeze(0).expand(B, -1)
            return out[mr].reshape(B, -1, D), tgt_jepa[mr].reshape(B, -1, D)

        pred_t, tgt_t = predict_region(self.text_cells)
        pred_p, tgt_p = predict_region(self.photo_cells)
        mse_loss = F.mse_loss(pred_t, tgt_t) + F.mse_loss(pred_p, tgt_p)
        loss = w_mse * mse_loss
        stats = {}

        # --- CLIP/SigLIP on ENHANCED features ---
        if w_clip > 0:
            photo_idx = self.photo_cells.to(dev)
            photo_global = F.normalize(ctx_enh[:, photo_idx].mean(dim=1), dim=-1)

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

            if align_mode == "sentence":
                # sentence-level: text global pooling vs photo global
                text_idx = self.text_cells.to(dev)
                text_global = F.normalize(ctx_enh[:, text_idx].mean(dim=1), dim=-1)
                # gather text across GPUs too
                text_all = text_global
                if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                    gathered_t = [torch.zeros_like(text_global) for _ in range(world)]
                    dist.all_gather(gathered_t, text_global.contiguous())
                    gathered_t[rank] = text_global
                    text_all = torch.cat(gathered_t, dim=0)
                sim = (text_all @ photo_all.T) * logit_scale  # (world*B, world*B)
                Nb = sim.size(0)
                labels = torch.eye(Nb, device=sim.device)
                sign = 2 * labels - 1
                clip_loss = -F.logsigmoid(sim * sign).mean()
            else:
                # word-level: each word vs photo
                wm = word_masks.to(dev).float()
                wv = word_valid.to(dev).float()
                word_feats = torch.einsum("bwn,bnd->bwd", wm, ctx_enh.float())
                denom = wm.sum(dim=2).clamp(min=1).unsqueeze(-1)
                word_feats = F.normalize(word_feats / denom, dim=-1)
                sim = torch.einsum("bwd,cd->bwc", word_feats, photo_all)
                Bm, max_w, Nb = sim.shape
                if loss_type == "siglip":
                    labels_2d = torch.zeros(Bm, Nb, device=sim.device)
                    diag = torch.arange(Bm, device=sim.device) + rank * Bm
                    labels_2d[torch.arange(Bm), diag] = 1.0
                    sign = (2 * labels_2d - 1).unsqueeze(1).expand(Bm, max_w, Nb)
                    clip_loss = (-F.logsigmoid(sim * logit_scale * sign).mean(2) * wv).sum() / wv.sum().clamp(min=1)
                else:
                    sim_scaled = sim * logit_scale
                    labels = torch.arange(Bm, device=dev) + rank * Bm
                    labels = labels.unsqueeze(1).expand(Bm, max_w)
                    ce = F.cross_entropy(sim_scaled.reshape(Bm * max_w, Nb), labels.reshape(Bm * max_w), reduction="none")
                    clip_loss = (ce.reshape(Bm, max_w) * wv).sum() / wv.sum().clamp(min=1)

            stats["clip"] = clip_loss.detach()
            loss = loss + w_clip * clip_loss

        # SIGReg on enhanced features
        if lam > 0:
            reg = self.sigreg(ctx_enh.transpose(0, 1))
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
    p.add_argument("--loss_type", type=str, default="siglip", choices=["siglip", "info_nce"])
    p.add_argument("--align_mode", type=str, default="sentence", choices=["sentence", "word"],
                   help="sentence=stable global alignment, word=per-word alignment")
    p.add_argument("--ctm_iters", type=int, default=50)
    p.add_argument("--ctm_memory", type=int, default=4)
    p.add_argument("--ctm_thoughts", type=int, default=8)
    p.add_argument("--bptt_window", type=int, default=15)
    p.add_argument("--grad_checkpoint", type=int, default=1)
    p.add_argument("--freeze_enhancer", type=int, default=0, help="1=freeze CTM enhancer params (preserve initial features)")
    p.add_argument("--unified_epochs", type=int, default=0,
                   help="epochs with UNIFIED gradient (JEPA through enhancer) before switching to split_grad; 0=always split_grad")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4, help="lr for predictor + other params")
    p.add_argument("--lr_encoder", type=float, default=2e-5, help="lr for backbone (low to protect)")
    p.add_argument("--lr_enhancer", type=float, default=2e-5, help="lr for CTM enhancer (try higher: 1e-4/2e-4)")
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--bf16", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="./outputs/ctm_enc_jepa")
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=10)
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

    model = CTMEncoderJepa("convnext", args.img_size, args.hidden, args.layers, args.heads,
                           args.patch_size, args.pred_depth, args.ema_tau,
                           args.ctm_iters, args.ctm_memory, args.ctm_thoughts,
                           args.bptt_window, freeze_enhancer=bool(args.freeze_enhancer)).to(device)
    base = model.module if isinstance(model, DDP) else model
    if hasattr(base.enhancer, "use_checkpoint"):
        base.enhancer.use_checkpoint = bool(args.grad_checkpoint)
    if world > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    base = model.module if isinstance(model, DDP) else model
    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[model] CTMEncoderJepa params(M)={n_params/1e6:.1f} "
              f"ctm_iters={args.ctm_iters} thoughts={args.ctm_thoughts}", flush=True)

    # parameter groups: backbone / enhancer / other (separate lr)
    backbone_ids = set(id(p) for p in base.backbone.parameters())
    enhancer_ids = set(id(p) for p in base.enhancer.parameters())
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) in backbone_ids]
    enhancer_params = [p for p in model.parameters() if p.requires_grad and id(p) in enhancer_ids]
    other_params = [p for p in model.parameters()
                    if p.requires_grad and id(p) not in backbone_ids and id(p) not in enhancer_ids]
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_encoder},
        {"params": enhancer_params, "lr": args.lr_enhancer},
        {"params": other_params, "lr": args.lr},
    ], weight_decay=args.wd)
    if is_main:
        print(f"[opt] backbone lr={args.lr_encoder} ({len(backbone_params)} params), "
              f"enhancer lr={args.lr_enhancer} ({len(enhancer_params)} params), "
              f"other lr={args.lr} ({len(other_params)} params)", flush=True)
    total_steps = args.epochs * len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total_steps, args.warmup))

    writer = None
    if is_main:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(out / "tb")
        print(f"[tb] tensorboard --logdir {out}/tb", flush=True)

    step = 0
    for epoch in range(args.epochs):
        if world > 1:
            sampler.set_epoch(epoch)
        model.train()
        split_grad = (epoch >= args.unified_epochs)
        if is_main and epoch == args.unified_epochs and args.unified_epochs > 0:
            print(f"[switch] epoch {epoch}: unified → split_grad", flush=True)
        if is_main:
            bar = tqdm(train, desc=f"e{epoch}({'sp' if split_grad else 'uni'})", dynamic_ncols=True, mininterval=2.0)
        else:
            bar = train
        for composite, word_masks, word_valid in bar:
            composite = composite.to(device, non_blocking=True)
            word_masks = word_masks.to(device, non_blocking=True)
            word_valid = word_valid.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp, dtype=torch.bfloat16):
                loss, stats = model(composite, word_masks, word_valid,
                                    lam=args.lam, w_mse=args.w_mse, w_clip=args.w_clip,
                                    loss_type=args.loss_type, align_mode=args.align_mode,
                                    split_grad=split_grad)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                if is_main:
                    print(f"[WARN] NaN/Inf grad at step {step}, skipping", flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            opt.step(); sched.step()
            base.update_ema(epoch, args.epochs)
            if is_main:
                if step % args.log_every == 0:
                    clip_val = float(stats.get("clip", torch.tensor(0.0)))
                    bar.set_postfix(loss=f"{loss.item():.4f}", cos=f"{float(stats['cos']):.3f}",
                                    clip=f"{clip_val:.3f}", lr=f"{sched.get_last_lr()[0]:.1e}")
                    if writer:
                        writer.add_scalar("train/loss", loss.item(), step)
                        writer.add_scalar("train/clip", clip_val, step)
                        writer.add_scalar("train/cos", float(stats["cos"]), step)
                        writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
                step += 1
        if is_main:
            do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs - 1)
            do_save = (epoch % args.save_every == 0) or (epoch == args.epochs - 1)
            if do_eval:
                syn, ant, rnd = semantic_eval(base, device, amp, args.img_size, base.grid, DEFAULT_FONT)
                print(f"== epoch {epoch} semantic: syn={syn:.3f} ant={ant:.3f} random={rnd:.3f} "
                      f"(syn-rand={syn-rnd:+.3f}, ant-rand={ant-rnd:+.3f}) ==", flush=True)
                if writer:
                    writer.add_scalar("eval/syn", syn, epoch)
                    writer.add_scalar("eval/ant", ant, epoch)
                    writer.add_scalar("eval/ant_minus_rand", ant - rnd, epoch)
            if do_save:
                torch.save({"model": base.state_dict(),
                            "args": {**vars(args), "arch": "convnext", "objective": "ctm_enc_jepa"}},
                           out / f"epoch{epoch}.pt")
                print(f"   [saved checkpoint at epoch {epoch}]", flush=True)
    if writer:
        writer.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
