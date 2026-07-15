"""CTM-JEPA: Cross-modal JEPA with a CTM-style predictor.

Replaces the 4-layer Transformer predictor with a Continuous Thought Machine
predictor that iteratively refines its prediction over T ticks. Each tick:
  1. Cross-attention (cells attend to each other)
  2. Synapse update (state + attention → new state)
  3. NLMs (per-neuron processing of memory trace, via einsum)

The encoder (ConvNeXt) and all other components (CLIP loss, SIGReg, EMA)
remain identical to train_crossmodal_jepa.py — only the predictor changes.

Key CTM ideas preserved:
  - Iterative reasoning (T ticks, not single forward)
  - Memory (M-step sliding trace per neuron)
  - Per-neuron dynamics (SuperLinear: each of D neurons has private MLP)

Input:  feature_map (B, N, D) + mask (B, N)
Output: predicted latents (B, N, D) — same interface as old predictor
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
from backbones import build_encoder
from train import setup_ddp, lr_lambda
from model import SIGReg

from train_crossmodal_jepa import (
    build_composite, boxes_to_cell_masks,
    TarImageText, make_collate, region_mask, semantic_eval,
)


class SuperLinear(nn.Module):
    """Neuron-Level Model (NLM) from CTM. Each of D neurons has a private
    linear layer mapping M-step history → 1 output. Implemented via einsum."""

    def __init__(self, memory_length, num_neurons):
        super().__init__()
        self.layernorm = nn.LayerNorm(memory_length)
        bound = 1.0 / math.sqrt(memory_length + 1)
        self.w = nn.Parameter(torch.empty(memory_length, 1, num_neurons).uniform_(-bound, bound))
        self.b = nn.Parameter(torch.zeros(1, num_neurons, 1))
        self.scale = nn.Parameter(torch.Tensor([0.0]))  # init 0: NLM starts inactive, learns gradually

    def forward(self, trace):
        # trace: (..., D, M) — each neuron's M-step history
        out = self.layernorm(trace)
        out = torch.einsum("...dm,mhd->...dh", out, self.w) + self.b  # (..., D, 1)
        return torch.tanh(out.squeeze(-1)) * self.scale  # (..., D) — tanh bounds output


class CTMPredictor(nn.Module):
    """CTM predictor with K thought tokens (K << N cells).

    Thought tokens iteratively attend to feature map over T ticks,
    then broadcast back to all N cells via output attention.
    Per-tick cost: O(K*N*D) instead of O(N^2*D), enabling T=50+ ticks.

    Each tick:
      1. K thoughts cross-attend to N feature-map cells
      2. Synapse update (thoughts + attn → new thoughts)
      3. NLM (per-neuron memory processing)
    Final: N cells cross-attend to K thoughts → predicted latents
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

        # thought token init (K learnable tokens)
        self.thought_init = nn.Parameter(torch.zeros(1, num_thoughts, dim))
        nn.init.normal_(self.thought_init, std=0.02)

        # mask token + position for feature map
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

        # per-tick: thoughts → feature_map attention
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.attn_out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)

        # synapse: concat(thoughts, attn_out) → new thoughts
        self.synapse = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.LayerNorm(dim))

        # NLM: per-neuron memory processing
        self.nlm = SuperLinear(memory_length, dim)
        self.nlm_norm = nn.LayerNorm(dim)

        # output broadcast: feature_map cells → thoughts
        self.out_q_proj = nn.Linear(dim, dim)
        self.out_kv_proj = nn.Linear(dim, dim * 2)
        self.out_attn_proj = nn.Linear(dim, dim)
        self.out_norm = nn.LayerNorm(dim)

    def _tick(self, thoughts, kv_input, trace):
        """One tick: K thoughts attend to N cells, update via synapse + NLM."""
        q = self.q_proj(thoughts)          # (B, K, D)
        kv = self.kv_proj(kv_input)        # (B, N, 2D)
        k, v = kv.chunk(2, dim=-1)         # (B, N, D) each
        B, K, D = q.shape
        N = k.size(1)
        hd = D // self.num_heads
        qh = q.reshape(B, K, self.num_heads, hd).transpose(1, 2)
        kh = k.reshape(B, N, self.num_heads, hd).transpose(1, 2)
        vh = v.reshape(B, N, self.num_heads, hd).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(qh, kh, vh)  # (B, H, K, hd)
        attn_out = attn_out.transpose(1, 2).reshape(B, K, D)
        attn_out = self.attn_out_proj(attn_out)
        thoughts = self.norm1(thoughts + attn_out)
        thoughts = self.synapse(torch.cat([thoughts, attn_out], dim=-1))

        trace = torch.cat([trace[..., 1:], thoughts.unsqueeze(-1)], dim=-1)
        nlm_out = self.nlm(trace)
        thoughts = thoughts + self.nlm_norm(nlm_out)
        return thoughts, trace

    def forward(self, feature_map, mask):
        """feature_map: (B,N,D), mask: (B,N) bool → predicted latents (B,N,D)."""
        B, N, D = feature_map.shape
        M = self.memory_length
        K = self.num_thoughts

        # prepare kv from feature map (mask replacement)
        x = feature_map + self.pos
        m = mask.unsqueeze(-1)
        x = torch.where(m, self.mask_token.expand(B, N, D) + self.pos, x)

        # init thoughts + trace
        thoughts = self.thought_init.expand(B, K, D)
        trace = thoughts.unsqueeze(-1).expand(B, K, D, M).contiguous()

        # T ticks of iterative reasoning (truncated BPTT for stability)
        for t in range(self.num_iters):
            if t > 0 and t % self.bptt_window == 0:
                thoughts = thoughts.detach()
                trace = trace.detach()
            if self.training and self.use_checkpoint:
                thoughts, trace = checkpoint(self._tick, thoughts, x, trace, use_reentrant=False)
            else:
                thoughts, trace = self._tick(thoughts, x, trace)

        # broadcast: N cells query K thoughts → predicted latents
        q = self.out_q_proj(x)              # (B, N, D)
        kv = self.out_kv_proj(thoughts)     # (B, K, 2D)
        k, v = kv.chunk(2, dim=-1)
        hd = D // self.num_heads
        qh = q.reshape(B, N, self.num_heads, hd).transpose(1, 2)
        kh = k.reshape(B, K, self.num_heads, hd).transpose(1, 2)
        vh = v.reshape(B, K, self.num_heads, hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(qh, kh, vh)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out_norm(x + self.out_attn_proj(out))
        return out


class CTMJepa(nn.Module):
    def __init__(self, arch="convnext", img_size=224, hidden=768, layers=12,
                 heads=12, patch=16, ema_tau=0.996, ctm_iters=50, ctm_memory=4,
                 ctm_thoughts=8):
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

        # CTM predictor (thought-token based, T ticks)
        self.predictor = CTMPredictor(
            self.feat_dim, g * g, num_iters=ctm_iters, memory_length=ctm_memory,
            num_thoughts=ctm_thoughts, num_heads=max(1, self.feat_dim // 64))
        self.pred_norm = nn.LayerNorm(self.feat_dim)

        self.target_encoder = self._copy(self.encoder)
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

        def predict_region(mask_region):
            out = self.pred_norm(self.predictor(ctx, mask_region.to(ctx.device).unsqueeze(0).expand(B, -1)))
            return out[mask_region.unsqueeze(0).expand(B, -1)].reshape(B, -1, D), \
                   tgt[mask_region.unsqueeze(0).expand(B, -1)].reshape(B, -1, D)

        pred_t, tgt_t = predict_region(self.text_cells)
        pred_p, tgt_p = predict_region(self.photo_cells)
        mse_loss = F.mse_loss(pred_t, tgt_t) + F.mse_loss(pred_p, tgt_p)
        loss = w_mse * mse_loss
        stats = {}

        # word-level CLIP (identical to cross-modal JEPA)
        if w_clip > 0 and word_masks is not None:
            wm = word_masks.to(ctx.device).float()
            wv = word_valid.to(ctx.device).float()
            word_feats = torch.einsum("bwn,bnd->bwd", wm, ctx.float())
            denom = wm.sum(dim=2).clamp(min=1).unsqueeze(-1)
            word_feats = F.normalize(word_feats / denom, dim=-1)
            photo_idx = self.photo_cells.to(ctx.device)
            photo_global = F.normalize(ctx[:, photo_idx].mean(dim=1), dim=-1)

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
    p.add_argument("--ema_tau", type=float, default=0.996)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--w_mse", type=float, default=0.3)
    p.add_argument("--w_clip", type=float, default=1.0)
    p.add_argument("--ctm_iters", type=int, default=50, help="CTM thinking ticks (T)")
    p.add_argument("--ctm_memory", type=int, default=4, help="CTM memory length (M)")
    p.add_argument("--ctm_thoughts", type=int, default=8, help="number of thought tokens (K)")
    p.add_argument("--bptt_window", type=int, default=15, help="truncated BPTT window size")
    p.add_argument("--grad_checkpoint", type=int, default=1, help="gradient checkpointing (1=on for T>20)")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--bf16", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="./outputs/ctm_jepa")
    p.add_argument("--save_every", type=int, default=50)
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

    model = CTMJepa("convnext", args.img_size, args.hidden, args.layers, args.heads,
                    args.patch_size, args.ema_tau, args.ctm_iters, args.ctm_memory,
                    args.ctm_thoughts).to(device)
    base = model.module if isinstance(model, DDP) else model
    base.predictor.bptt_window = args.bptt_window
    base.predictor.use_checkpoint = bool(args.grad_checkpoint)
    if world > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    base = model.module if isinstance(model, DDP) else model
    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[model] CTMJepa params(M)={n_params/1e6:.1f} "
              f"ctm_iters={args.ctm_iters} memory={args.ctm_memory}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total_steps, args.warmup))

    # tensorboard
    writer = None
    if is_main:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(out / "tb")
        print(f"[tb] logs at {out}/tb — run: tensorboard --logdir {out}/tb", flush=True)

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
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                if is_main:
                    print(f"[WARN] NaN/Inf grad at step {step}, skipping update", flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            opt.step(); sched.step()
            base.update_ema()
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
                        if "reg" in stats:
                            writer.add_scalar("train/sigreg", float(stats["reg"]), step)
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
                    writer.add_scalar("eval/random", rnd, epoch)
                    writer.add_scalar("eval/syn_minus_rand", syn - rnd, epoch)
                    writer.add_scalar("eval/ant_minus_rand", ant - rnd, epoch)
            if do_save:
                torch.save({"model": base.state_dict(),
                            "args": {**vars(args), "arch": "convnext", "objective": "ctm_jepa"}},
                           out / f"epoch{epoch}.pt")
                print(f"   [saved checkpoint at epoch {epoch}]", flush=True)
    if writer:
        writer.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
