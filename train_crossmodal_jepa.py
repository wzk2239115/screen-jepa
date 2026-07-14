"""Cross-modal JEPA on text+photo composites.

Composite image = [rendered caption (top half) | photo (bottom half)] in one
224x224 canvas. One convnext encodes the composite -> 14x14 feature map (top
rows = text region, bottom rows = photo region). JEPA objective: mask one
region's features (replace with [MASK] token) and predict its latent (from the
EMA target encoder on the full composite) using the OTHER region as context.
Symmetric (predict photo<-text AND text<-photo).

Predicting the photo latent from the caption forces the text features to encode
the caption's meaning -> semantic structure emerges from the image-text grounding.
Satisfies: JEPA (masked latent prediction) + composite (text+photo in one image).
"""
import argparse
import io
import os
import random
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from render import DEFAULT_FONT
from backbones import build_encoder, TransformerBlock
from train import setup_ddp, lr_lambda
from model import SIGReg
from probe import ANTONYMS, SYNONYMS

W, H = 224, 112  # each half of the 224x224 composite


def render_caption_block(caption, ww=W, hh=H, font_path=DEFAULT_FONT):
    """Render a caption auto-fit into a ww x hh white block (black text).
    Returns (img, boxes) where boxes is a list of (x0,y0,x1,y1) per word."""
    img = Image.new("RGB", (ww, hh), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    words = caption.split()
    lo, hi, best = 8, 36, 8
    best_font, best_pl = None, None
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(font_path, mid)
        lh = int(round(mid * 1.25))
        sp = font.getlength(" ")
        pl, y, x, ok = [], 0, 6, True
        usable_w, usable_h = ww - 6, hh - 4
        for w in words:
            wl = font.getlength(w)
            if x + wl > usable_w and x > 6:
                x = 6; y += lh
            if y + lh > usable_h:
                ok = False; break
            pl.append((w, x, y, x + wl, y + lh)); x += wl + sp
        if ok:
            best, best_font, best_pl = mid, font, pl; lo = mid + 1
        else:
            hi = mid - 1
    if best_pl is None:
        best_font = ImageFont.truetype(font_path, best); best_pl = []
    boxes = []
    for w, x, y, x1, y1 in best_pl:
        draw.text((x, y), w, fill=(0, 0, 0), font=best_font)
        boxes.append((int(x), int(y), int(x1), int(y1)))
    return np.array(img), boxes


def build_composite(caption, photo_pil, font_path=DEFAULT_FONT):
    text_block, boxes = render_caption_block(caption, font_path=font_path)
    photo = photo_pil.convert("RGB").resize((W, H))
    composite = np.concatenate([text_block, np.array(photo)], axis=0)  # 224x224
    return composite, boxes


def boxes_to_cell_masks(boxes, grid, patch_size):
    """Convert per-word pixel bboxes to per-word cell masks (grid*grid bool each).
    Bbox coords are in the 224x224 composite frame (text occupies top half)."""
    g, ps, N = grid, patch_size, grid * grid
    if not boxes:
        m = torch.zeros(N, dtype=torch.bool)
        m[: g * (g // 2)] = True
        return m.unsqueeze(0)
    masks = []
    for x0, y0, x1, y1 in boxes:
        r0 = max(0, y0 // ps); r1 = min(g - 1, max(r0, (y1 - 1) // ps))
        c0 = max(0, x0 // ps); c1 = min(g - 1, max(c0, (x1 - 1) // ps))
        m = torch.zeros(N, dtype=torch.bool)
        for rr in range(r0, r1 + 1):
            base = rr * g
            for cc in range(c0, c1 + 1):
                m[base + cc] = True
        masks.append(m)
    return torch.stack(masks)


_TAR_CACHE = {}


def _tar(path):
    if path not in _TAR_CACHE:
        _TAR_CACHE[path] = tarfile.open(path)
    return _TAR_CACHE[path]


class TarImageText(Dataset):
    def __init__(self, tar_dir, num_tars=None, img_size=224, font_path=DEFAULT_FONT,
                 grid=14, patch_size=16):
        tars = sorted([str(p) for p in Path(tar_dir).glob("*.tar")])
        if num_tars:
            tars = tars[:num_tars]
        self.index = []
        import json
        self.json = json
        good_tars = 0
        for tp in tars:
            try:
                tf = tarfile.open(tp)
                members = tf.getmembers()
                for m in members:
                    if m.name.endswith(".jpg"):
                        self.index.append((tp, m.name))
                tf.close()
                good_tars += 1
            except Exception as e:
                print(f"[data] WARNING: skipping corrupt tar {tp}: {e}", flush=True)
        self.font_path = font_path
        self.img_size = img_size
        self.grid = grid
        self.patch_size = patch_size
        print(f"[data] indexed {len(self.index)} image-text pairs from {good_tars}/{len(tars)} tars", flush=True)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        for _ in range(5):
            try:
                tp, name = self.index[random.randint(0, len(self.index) - 1)]
                tf = _tar(tp)
                img = Image.open(io.BytesIO(tf.extractfile(name).read()))
                cap = self.json.loads(tf.extractfile(name.replace(".jpg", ".json")).read())["caption"]
                composite, boxes = build_composite(cap, img, self.font_path)
                t = torch.from_numpy(composite).float().permute(2, 0, 1) / 255.0
                masks = boxes_to_cell_masks(boxes, self.grid, self.patch_size)
                return (t - 0.5) / 0.5, masks, masks.size(0)
            except Exception:
                continue
        # all retries failed: return a blank
        t = torch.zeros(3, 224, 224)
        m = torch.zeros(self.grid * self.grid, dtype=torch.bool).unsqueeze(0)
        return t, m, 1


def make_collate(grid):
    """Collate variable-length word masks into padded (B, max_w, N) tensors."""
    N = grid * grid

    def collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        max_w = max(b[2] for b in batch)
        B = len(batch)
        M = torch.zeros(B, max_w, N, dtype=torch.bool)
        valid = torch.zeros(B, max_w, dtype=torch.bool)
        for bi, b in enumerate(batch):
            n = b[2]
            M[bi, :n] = b[1]
            valid[bi, :n] = True
        return imgs, M, valid

    return collate


def region_mask(grid, rows):
    m = torch.zeros(grid, grid, dtype=torch.bool)
    m[rows, :] = True
    return m.flatten()


class CrossModalJEPA(nn.Module):
    def __init__(self, arch="convnext", img_size=224, hidden=768, layers=12,
                 heads=12, patch=16, pred_depth=4, ema_tau=0.996):
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
            *[TransformerBlock(self.feat_dim, heads=max(1, self.feat_dim // 64)) for _ in range(pred_depth)])
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

    def init_from(self, ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck["model"]
        enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        self.encoder.load_state_dict(enc, strict=False)
        self.target_encoder.load_state_dict(enc, strict=False)

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

        # word-level CLIP alignment: each word feat <-> own photo feat (InfoNCE)
        if w_clip > 0 and word_masks is not None:
            wm = word_masks.to(ctx.device).float()  # (B, max_w, N)
            wv = word_valid.to(ctx.device).float()  # (B, max_w)
            word_feats = torch.einsum("bwn,bnd->bwd", wm, ctx.float())  # (B, max_w, D)
            denom = wm.sum(dim=2).clamp(min=1).unsqueeze(-1)
            word_feats = F.normalize(word_feats / denom, dim=-1)
            photo_idx = self.photo_cells.to(ctx.device)
            photo_global = F.normalize(ctx[:, photo_idx].mean(dim=1), dim=-1)  # (B, D)

            # gather photo_global across GPUs for larger negative pool
            rank = 0
            photo_all = photo_global
            if dist.is_available() and dist.is_initialized():
                world = dist.get_world_size()
                if world > 1:
                    gathered = [torch.zeros_like(photo_global) for _ in range(world)]
                    dist.all_gather(gathered, photo_global.contiguous())
                    rank = dist.get_rank()
                    gathered[rank] = photo_global  # keep local gradient
                    photo_all = torch.cat(gathered, dim=0)  # (world*B, D)

            logit_scale = self.logit_scale.exp().clamp(max=100.0)
            sim = torch.einsum("bwd,cd->bwc", word_feats, photo_all) * logit_scale  # (B, max_w, world*B)
            Bm, max_w, Nb = sim.shape
            labels = torch.arange(Bm, device=ctx.device) + rank * Bm
            labels = labels.unsqueeze(1).expand(Bm, max_w)
            ce = F.cross_entropy(sim.reshape(Bm * max_w, Nb), labels.reshape(Bm * max_w), reduction="none")
            clip_loss = (ce.reshape(Bm, max_w) * wv).sum() / wv.sum().clamp(min=1)
            stats["clip"] = clip_loss.detach()
            loss = loss + w_clip * clip_loss

        if lam > 0:
            reg = self.sigreg(ctx.transpose(0, 1))  # (N, B, D) as lewm
            loss = loss + lam * reg
            stats["reg"] = reg.detach()
        with torch.no_grad():
            stats["cos"] = 0.5 * (F.cosine_similarity(pred_t, tgt_t, dim=-1).mean()
                                  + F.cosine_similarity(pred_p, tgt_p, dim=-1).mean())
        return loss, stats


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tar_dir", default="/home/wzk/datasets/recap-datacomp-384-1M")
    p.add_argument("--num_tars", type=int, default=5)
    p.add_argument("--init_from", default=None)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--pred_depth", type=int, default=4)
    p.add_argument("--ema_tau", type=float, default=0.996)
    p.add_argument("--lam", type=float, default=0.1, help="VICReg anti-collapse weight")
    p.add_argument("--w_mse", type=float, default=0.3, help="weight on latent-prediction MSE")
    p.add_argument("--w_clip", type=float, default=1.0, help="weight on word-level CLIP InfoNCE")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--bf16", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="./outputs/xmodal")
    p.add_argument("--save_every", type=int, default=50, help="save checkpoint every N epochs")
    p.add_argument("--eval_every", type=int, default=10, help="semantic eval every N epochs")
    return p.parse_args()


@torch.no_grad()
def semantic_eval(model, device, amp, img_size, grid, font_path):
    half = img_size // 2
    model.eval()
    g = grid

    def emb(word):
        xs = []
        for fs in (56, 72, 88):
            block = Image.new("RGB", (img_size, half), (255, 255, 255))
            d = ImageDraw.Draw(block)
            font = ImageFont.truetype(font_path, fs)
            d.text((img_size / 2, half / 2), word, fill=(0, 0, 0), font=font, anchor="mm")
            comp = np.concatenate([np.array(block), np.full((half, img_size, 3), 255, dtype=np.uint8)], axis=0)
            t = torch.from_numpy(comp).float().permute(2, 0, 1) / 255.0
            xs.append((t - 0.5) / 0.5)
        with torch.autocast("cuda", enabled=amp, dtype=torch.bfloat16):
            f = model.encoder(torch.stack(xs).to(device), return_map=True).float()
        text_cells = region_mask(g, list(range(g // 2))).to(device)
        z = f[:, text_cells].mean(dim=1)
        return F.normalize(z.mean(0), dim=0)

    words = sorted({w for pr in SYNONYMS + ANTONYMS for w in pr})
    E = {w: emb(w) for w in words}

    def avg(pairs):
        vals = [float((E[a] * E[b]).sum()) for a, b in pairs if a in E and b in E]
        return float(np.mean(vals)) if vals else 0.0

    syn, ant = avg(SYNONYMS), avg(ANTONYMS)
    wl = list(E.values())
    rnd = float(np.mean([float((wl[i] * wl[j]).sum()) for i, j in
                         [random.sample(range(len(wl)), 2) for _ in range(160)]]))
    model.train()
    return syn, ant, rnd


def main():
    args = build_args()
    amp = bool(args.bf16)
    rank, local_rank, world = setup_ddp()
    is_main = rank == 0
    device = f"cuda:{local_rank}"
    if is_main:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
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

    model = CrossModalJEPA("convnext", args.img_size, args.hidden, args.layers, args.heads,
                           args.patch_size, args.pred_depth, args.ema_tau).to(device)
    if args.init_from:
        model.init_from(args.init_from)
        if is_main:
            print(f"[xmodal] loaded encoder from {args.init_from}", flush=True)
    if world > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    base = model.module if isinstance(model, DDP) else model
    if is_main:
        print(f"[model] trainable params(M)={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total_steps, args.warmup))

    step = 0
    for epoch in range(args.epochs):
        if world > 1:
            sampler.set_epoch(epoch)
        model.train()
        bar = tqdm(train, desc=f"e{epoch}", disable=not is_main, dynamic_ncols=True, mininterval=2.0)
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
                                clip=f"{clip_val:.3f}",
                                lr=f"{sched.get_last_lr()[0]:.1e}")
            step += 1
        if is_main:
            do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs - 1)
            do_save = (epoch % args.save_every == 0) or (epoch == args.epochs - 1)
            if do_eval:
                syn, ant, rnd = semantic_eval(base, device, amp, args.img_size, base.grid, DEFAULT_FONT)
                print(f"== epoch {epoch} semantic: syn={syn:.3f} ant={ant:.3f} random={rnd:.3f} "
                      f"(syn-rand={syn-rnd:+.3f}, ant-rand={ant-rnd:+.3f}) ==", flush=True)
            if do_save:
                torch.save({"model": base.state_dict(),
                            "args": {**vars(args), "arch": "convnext", "objective": "xmodal"}},
                           out / f"epoch{epoch}.pt")
                print(f"   [saved checkpoint at epoch {epoch}]", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
