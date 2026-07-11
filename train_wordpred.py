"""Stage-3: masked-WORD prediction with pixel input (MLM, pixel edition).

Take the Stage-1 convnext encoder (from aug_L1), UNFREEZE it, add a small
classification head that predicts the identity of the masked word from the
masked-position features. Because we render the text, we know every word ->
free labels. The word-prediction gradient flows back into the encoder, pressing
distributional/semantic structure into it.

Usage:
    python train_wordpred.py --init_from outputs/aug_L1/epoch49.pt \
        --parquet_path "$PARQUET" --language English ...
"""
import argparse
import collections
import csv
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, random_split
from tqdm import tqdm

from dataset import load_sentences, to_tensor
from render import TextRenderer, patchwork_bg
from backbones import build_encoder
from train import setup_ddp, lr_lambda


def build_vocab(sentences, top_k=10000):
    freq = collections.Counter()
    for s in sentences:
        for w in s.lower().split():
            freq[w] += 1
    most = [w for w, _ in freq.most_common(top_k)]
    w2i = {w: i for i, w in enumerate(most)}
    return w2i, most


class WordPredDataset(torch.utils.data.Dataset):
    """Mask ONE in-vocab word per sample; return (full, masked, cell_mask, word_idx)."""

    def __init__(self, sentences, w2i, img_size=224, font_size=72, mask_ratio=0.2,
                 grid=14, bg_block=True, font_augment=True, font_pool=None):
        self.sents = sentences
        self.w2i = w2i
        self.grid = grid
        self.stride = img_size / grid
        self.bg_block = bg_block
        from dataset import DEFAULT_FONTS, DEFAULT_FONT
        pool = (font_pool.split(",") if font_pool else DEFAULT_FONTS)
        import os as _os
        pool = [p for p in pool if _os.path.exists(p)] or [DEFAULT_FONT]
        pl = pool if font_augment else [pool[0]]
        self.renderers = [TextRenderer(img_size=img_size, font_size=font_size, font_path=p) for p in pl]

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, i):
        sent = self.sents[i]
        words = sent.split()
        # candidate in-vocab word indices in this sentence
        cands = [j for j, w in enumerate(words) if w.lower() in self.w2i]
        if not cands:
            r = self.renderers[0]
            full = np.full((r.img_size, r.img_size, 3), 255, dtype=np.uint8)
            t = to_tensor(full)
            return t, t.clone(), torch.zeros(self.grid * self.grid, dtype=torch.bool), 0
        ti = cands[np.random.randint(len(cands))]
        target_word = words[ti].lower()
        r = self.renderers[np.random.randint(len(self.renderers))]
        S = r.img_size
        bg_img = patchwork_bg(S) if self.bg_block else None
        bg = (230, 230, 230) if bg_img is not None else (255, 255, 255)
        full, boxes = r.render(sent, bg_color=bg, bg_img=bg_img)
        if ti >= len(boxes):
            full = np.full((S, S, 3), 255, dtype=np.uint8)
            t = to_tensor(full)
            return t, t.clone(), torch.zeros(self.grid * self.grid, dtype=torch.bool), 0
        masked = r.mask_words(full, boxes, [ti], bg_color=bg, bg_img=bg_img)
        x0, y0, x1, y1 = boxes[ti]
        g = self.grid
        cm = torch.zeros(g, g, dtype=torch.bool)
        c0 = int(min(max(x0 / self.stride, 0), g - 1))
        c1 = int(min(max((x1 - 1) / self.stride, 0), g - 1))
        r0 = int(min(max(y0 / self.stride, 0), g - 1))
        r1 = int(min(max((y1 - 1) / self.stride, 0), g - 1))
        cm[r0:r1 + 1, c0:c1 + 1] = True
        return to_tensor(full), to_tensor(masked), cm.flatten(), self.w2i[target_word]


class WordPred(nn.Module):
    """convnext encoder (unfrozen) + head: pool masked-cell features -> vocab logits."""

    def __init__(self, arch="convnext", img_size=224, hidden=768, layers=12,
                 heads=12, patch=16, vocab_size=10000, mlp_dim=2048):
        super().__init__()
        self.encoder = build_encoder(arch, img_size=img_size, dim=hidden,
                                     patch=patch, depth=layers, heads=heads)
        self.feat_dim = getattr(self.encoder, "feature_dim", self.encoder.out_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, mlp_dim), nn.GELU(),
            nn.Linear(mlp_dim, vocab_size))

    def init_from(self, ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck["model"]
        enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        self.encoder.load_state_dict(enc, strict=False)

    def forward(self, masked, cell_mask):
        feat = self.encoder(masked, return_map=True)  # (B, N, d)
        m = cell_mask.unsqueeze(-1).float()
        pooled = (feat * m).sum(1) / m.sum(1).clamp(min=1)  # (B, d)
        return self.head(pooled)

    def encode(self, x):
        return self.encoder(x, return_map=True).mean(dim=1)


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init_from", default=None)
    p.add_argument("--corpus_txt", default=None)
    p.add_argument("--parquet_path", required=True)
    p.add_argument("--language", default="English")
    p.add_argument("--ascii_only", type=int, default=1)
    p.add_argument("--max_sentences", type=int, default=1700000)
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--font_size", type=int, default=72)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--mlp_dim", type=int, default=2048)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--bf16", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.03)
    p.add_argument("--out", default="./outputs/wordpred")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    correct = top5 = total = 0
    for full, masked, cell_mask, word_idx in loader:
        masked = masked.to(device); cell_mask = cell_mask.to(device); word_idx = word_idx.to(device)
        with torch.autocast("cuda", enabled=amp, dtype=torch.bfloat16):
            logits = model(masked, cell_mask).float()
        pred = logits.argmax(-1)
        correct += (pred == word_idx).sum().item()
        top5 += (logits.topk(5, dim=1).indices == word_idx.unsqueeze(1)).any(1).sum().item()
        total += word_idx.size(0)
    model.train()
    return correct / max(1, total), top5 / max(1, total)


def main():
    args = build_args()
    amp = bool(args.bf16)
    rank, local_rank, world = setup_ddp()
    is_main = rank == 0
    device = f"cuda:{local_rank}"
    if is_main:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    if is_main:
        used = []
        sents = load_sentences(parquet_path=args.parquet_path, max_sentences=args.max_sentences,
                               ascii_only=bool(args.ascii_only), language=args.language, used=used)
        w2i, most = build_vocab(sents, args.vocab_size)
        (out / "datasource.txt").write_text("\n".join(used))
        (out / "vocab.txt").write_text("\n".join(most))
        print(f"[data] {len(sents)} sentences, vocab={len(w2i)}", flush=True)
    else:
        sents = w2i = most = None
    if world > 1:
        obj = [sents, w2i]; dist.broadcast_object_list(obj, src=0); sents, w2i = obj

    g = torch.Generator().manual_seed(args.seed)
    n_val = max(2000, int(len(sents) * args.val_frac))
    full_ds = WordPredDataset(sents, w2i, args.img_size, args.font_size,
                              grid=args.hidden and 14, bg_block=True, font_augment=True)
    train_ds, val_ds = random_split(full_ds, [len(sents) - n_val, n_val], generator=g)
    if world > 1:
        sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        train = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                           num_workers=args.workers, drop_last=True,
                           persistent_workers=args.workers > 0, pin_memory=True)
    else:
        train = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                           num_workers=args.workers, drop_last=True,
                           persistent_workers=args.workers > 0, pin_memory=True)
    val = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)

    model = WordPred("convnext", args.img_size, args.hidden, args.layers, args.heads,
                     args.patch_size, len(w2i), args.mlp_dim).to(device)
    if args.init_from:
        model.init_from(args.init_from)
        if is_main:
            print(f"[stage3] loaded encoder from {args.init_from} (unfrozen)", flush=True)
    if world > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    base = model.module if isinstance(model, DDP) else model
    if is_main:
        print(f"[model] trainable params(M)={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total_steps, args.warmup))

    step = 0
    for epoch in range(args.epochs):
        if world > 1:
            sampler.set_epoch(epoch)
        model.train()
        bar = tqdm(train, desc=f"e{epoch}", disable=not is_main, dynamic_ncols=True, mininterval=2.0)
        for full, masked, cell_mask, word_idx in bar:
            masked = masked.to(device, non_blocking=True)
            cell_mask = cell_mask.to(device, non_blocking=True)
            word_idx = word_idx.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp, dtype=torch.bfloat16):
                logits = model(masked, cell_mask)
                loss = F.cross_entropy(logits, word_idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); sched.step()
            if is_main and step % args.log_every == 0:
                acc = (logits.argmax(-1) == word_idx).float().mean().item()
                bar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{acc:.3f}",
                                lr=f"{sched.get_last_lr()[0]:.1e}")
            step += 1
        if is_main:
            t1, t5 = evaluate(base, val, device, amp)
            print(f"== epoch {epoch} val: top1={t1:.3f} top5={t5:.3f} (vocab={len(w2i)}, chance={1/len(w2i):.5f}) ==", flush=True)
            torch.save({"model": base.state_dict(),
                        "args": {**vars(args), "arch": "convnext", "objective": "wordpred"},
                        "vocab": most},
                       out / f"epoch{epoch}.pt")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
