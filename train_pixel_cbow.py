"""Pixel-CBOW / pixel-skip-gram: learn DISTRIBUTAL word vectors from pixels.

The convnext encoder plays the role of the word2vec lookup table: it maps a
word's IMAGE to a vector. Train with a skip-gram / CLIP-style InfoNCE so that a
center word's vector aligns with its context words' vectors. Because synonyms
share contexts, their vectors get pulled together -> semantic structure emerges
naturally from co-occurrence, no text teacher, pure pixels.

Usage:
    python train_pixel_cbow.py --init_from outputs/wordpred/epoch14.pt \
        --parquet_path "$PARQUET" --language English ...
"""
import argparse
import collections
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from dataset import DEFAULT_FONTS, DEFAULT_FONT, load_sentences, to_tensor
from render import TextRenderer, patchwork_bg
from backbones import build_encoder
from train import setup_ddp, lr_lambda
from probe import ANTONYMS, SYNONYMS


def build_vocab(sentences, top_k=10000):
    freq = collections.Counter()
    for s in sentences:
        for w in s.lower().split():
            freq[w] += 1
    most = [w for w, _ in freq.most_common(top_k)]
    return {w: i for i, w in enumerate(most)}, most


class CBOWPairs(Dataset):
    """On-the-fly (center, context) in-vocab word pairs, each rendered standalone."""

    def __init__(self, sentences, w2i, img_size=224, font_size=72, window=5,
                 pairs_per_epoch=4_000_000, bg_block=True, font_augment=True, font_pool=None):
        self.w2i = w2i
        self.i2w = {i: w for w, i in w2i.items()}
        self.window = window
        self.pairs_per_epoch = pairs_per_epoch
        self.sent_words = []
        for s in sentences:
            ws = [w2i[w] for w in s.lower().split() if w in w2i]
            if len(ws) >= 2:
                self.sent_words.append(ws)
        pool = (font_pool.split(",") if font_pool else DEFAULT_FONTS)
        pool = [p for p in pool if os.path.exists(p)] or [DEFAULT_FONT]
        pl = pool if font_augment else [pool[0]]
        self.renderers = [TextRenderer(img_size=img_size, font_size=font_size, font_path=p) for p in pl]
        self.bg_block = bg_block

    def __len__(self):
        return self.pairs_per_epoch

    def _render_word(self, w):
        r = self.renderers[random.randint(0, len(self.renderers) - 1)]
        S = r.img_size
        if self.bg_block:
            img, _ = r.render(w, bg_img=patchwork_bg(S))
        else:
            img, _ = r.render(w)
        return to_tensor(img)

    def __getitem__(self, _):
        ws = self.sent_words[random.randint(0, len(self.sent_words) - 1)]
        i = random.randint(0, len(ws) - 1)
        lo, hi = max(0, i - self.window), min(len(ws), i + self.window + 1)
        js = [j for j in range(lo, hi) if j != i]
        if not js:
            i2 = (i + 1) % len(ws)
            c, o = ws[i], ws[i2]
        else:
            c, o = ws[i], ws[random.choice(js)]
        return self._render_word(self.i2w[c]), self._render_word(self.i2w[o])


class PixelCBOW(nn.Module):
    def __init__(self, arch="convnext", img_size=224, hidden=768, layers=12,
                 heads=12, patch=16, proj_dim=256):
        super().__init__()
        self.encoder = build_encoder(arch, img_size=img_size, dim=hidden,
                                     patch=patch, depth=layers, heads=heads)
        self.feat_dim = getattr(self.encoder, "feature_dim", self.encoder.out_dim)
        self.proj = nn.Linear(self.feat_dim, proj_dim)

    def init_from(self, ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck["model"]
        enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        self.encoder.load_state_dict(enc, strict=False)

    def encode(self, x):
        f = self.encoder(x, return_map=True).mean(dim=1)
        return self.proj(f)

    def forward(self, center, context):
        return F.normalize(self.encode(center), dim=-1), F.normalize(self.encode(context), dim=-1)


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init_from", default=None)
    p.add_argument("--corpus_txt", default=None)
    p.add_argument("--parquet_path", required=True)
    p.add_argument("--language", default="English")
    p.add_argument("--ascii_only", type=int, default=1)
    p.add_argument("--max_sentences", type=int, default=1700000)
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--pairs_per_epoch", type=int, default=4_000_000)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--font_size", type=int, default=72)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--bf16", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="./outputs/cbow")
    return p.parse_args()


@torch.no_grad()
def semantic_eval(model, device, amp, img_size, font_size, n_aug=4):
    """Encode syn/ant words with the pixel encoder; report syn/ant/random cosine."""
    from dataset import DEFAULT_FONTS
    renderers = [TextRenderer(img_size=img_size, font_size=font_size, font_path=p)
                 for p in DEFAULT_FONTS if os.path.exists(p)][:4]
    model.eval()

    def emb(word):
        xs = []
        for r in renderers[:n_aug]:
            img, _ = r.render(word, bg_img=patchwork_bg(img_size))
            xs.append(to_tensor(img))
        with torch.autocast("cuda", enabled=amp, dtype=torch.bfloat16):
            z = model.encode(torch.stack(xs).to(device)).float()
        z = F.normalize(z.mean(0), dim=0)
        return z

    words = sorted({w for pr in SYNONYMS + ANTONYMS for w in pr})
    E = {w: emb(w) for w in words}

    def avg(pairs):
        return float(np.mean([float((E[a] * E[b]).sum()) for a, b in pairs
                               if a in E and b in E]))

    syn = avg(SYNONYMS)
    ant = avg(ANTONYMS)
    rnd = []
    wl = list(E.values())
    for _ in range(160):
        i, j = random.sample(range(len(wl)), 2)
        rnd.append(float((wl[i] * wl[j]).sum()))
    rnd = float(np.mean(rnd))
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

    ds = CBOWPairs(sents, w2i, args.img_size, args.font_size, args.window,
                   args.pairs_per_epoch)
    if world > 1:
        sampler = DistributedSampler(ds, shuffle=True, drop_last=True)
        train = DataLoader(ds, batch_size=args.batch, sampler=sampler,
                           num_workers=args.workers, drop_last=True,
                           persistent_workers=args.workers > 0, pin_memory=True)
    else:
        train = DataLoader(ds, batch_size=args.batch, shuffle=True,
                           num_workers=args.workers, drop_last=True,
                           persistent_workers=args.workers > 0, pin_memory=True)

    model = PixelCBOW("convnext", args.img_size, args.hidden, args.layers, args.heads,
                      args.patch_size, args.proj_dim).to(device)
    if args.init_from:
        model.init_from(args.init_from)
        if is_main:
            print(f"[cbow] loaded encoder from {args.init_from}", flush=True)
    if world > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    base = model.module if isinstance(model, DDP) else model
    if is_main:
        print(f"[model] params(M)={sum(p.numel() for p in model.parameters())/1e6:.1f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, total_steps, args.warmup))

    def step_loss(v_c, v_o):
        logits = v_c @ v_o.t() / args.tau
        labels = torch.arange(v_c.size(0), device=v_c.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    step = 0
    for epoch in range(args.epochs):
        if world > 1:
            sampler.set_epoch(epoch)
        model.train()
        bar = tqdm(train, desc=f"e{epoch}", disable=not is_main, dynamic_ncols=True, mininterval=2.0)
        for center, context in bar:
            center = center.to(device, non_blocking=True)
            context = context.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp, dtype=torch.bfloat16):
                v_c, v_o = model(center, context)
                loss = step_loss(v_c, v_o)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); sched.step()
            if is_main and step % args.log_every == 0:
                with torch.no_grad():
                    acc = (v_c @ v_o.t()).argmax(1) == torch.arange(v_c.size(0), device=device)
                    acc = acc.float().mean().item()
                bar.set_postfix(loss=f"{loss.item():.3f}", ctr_acc=f"{acc:.3f}",
                                lr=f"{sched.get_last_lr()[0]:.1e}")
            step += 1
        if is_main:
            syn, ant, rnd = semantic_eval(base, device, amp, args.img_size, args.font_size)
            print(f"== epoch {epoch} semantic: syn={syn:.3f} ant={ant:.3f} random={rnd:.3f} "
                  f"(syn-rand={syn-rnd:+.3f}, ant-rand={ant-rnd:+.3f}) ==", flush=True)
            torch.save({"model": base.state_dict(),
                        "args": {**vars(args), "arch": "convnext", "objective": "cbow"},
                        "vocab": most}, out / f"epoch{epoch}.pt")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
