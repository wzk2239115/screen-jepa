"""CLIP baseline on recap-datacomp-384-1M.

Fair comparison with train_crossmodal_jepa.py:
  - SAME image encoder (ConvNeXt)
  - SAME data, SAME training budget
  - ONLY difference: text is tokenized (BPE) + text transformer,
    NOT rendered as pixels

Standard CLIP: image encoder + text encoder + InfoNCE contrastive loss.
"""
import argparse
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from backbones import build_encoder
from train import setup_ddp, lr_lambda

# === tokenizer (local offline copy) ===
from transformers import CLIPTokenizerFast
_TOK_DIR = str(Path(__file__).parent / "clip_tokenizer")
_tok = CLIPTokenizerFast.from_pretrained(_TOK_DIR)
VOCAB_SIZE = _tok.vocab_size
BOS = _tok.bos_token_id
EOS = _tok.eos_token_id
PAD = _tok.pad_token_id or EOS
print(f"[clip] loaded CLIPTokenizer from {_TOK_DIR}: vocab={VOCAB_SIZE}", flush=True)


def tokenize(text, max_len=77):
    ids = _tok(text, padding="max_length", truncation=True,
               max_length=max_len, return_tensors="pt")["input_ids"][0]
    return ids


_TAR_CACHE = {}


def _tar(path):
    if path not in _TAR_CACHE:
        _TAR_CACHE[path] = tarfile.open(path)
    return _TAR_CACHE[path]


class TarImageTextCLIP(Dataset):
    def __init__(self, tar_dir, num_tars=None, img_size=224, max_len=77):
        tars = sorted([str(p) for p in Path(tar_dir).glob("*.tar")])
        if num_tars:
            tars = tars[:num_tars]
        self.index = []
        self.json = json
        good = 0
        for tp in tars:
            try:
                tf = tarfile.open(tp)
                for m in tf.getmembers():
                    if m.name.endswith(".jpg"):
                        self.index.append((tp, m.name))
                tf.close()
                good += 1
            except Exception as e:
                print(f"[data] skip corrupt tar {tp}: {e}", flush=True)
        self.img_size = img_size
        self.max_len = max_len
        print(f"[data] indexed {len(self.index)} pairs from {good}/{len(tars)} tars", flush=True)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        import random
        for _ in range(5):
            try:
                tp, name = self.index[random.randint(0, len(self.index) - 1)]
                tf = _tar(tp)
                img = Image.open(io.BytesIO(tf.extractfile(name).read()))
                cap = self.json.loads(tf.extractfile(name.replace(".jpg", ".json")).read())["caption"]
                img = img.convert("RGB").resize((self.img_size, self.img_size))
                t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
                t = (t - 0.5) / 0.5
                tokens = tokenize(cap, self.max_len)
                return t, tokens
            except Exception:
                continue
        return torch.zeros(3, self.img_size, self.img_size), tokenize("a photo", self.max_len)


class TextEncoder(nn.Module):
    """CLIP-style causal text transformer. Output = hidden at EOS position."""

    def __init__(self, vocab_size=49408, hidden=768, layers=8, heads=12, max_len=77):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden, padding_idx=PAD)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, hidden))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 4, dropout=0.0,
            batch_first=True, activation="gelu", norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)
        self.max_len = max_len

    def forward(self, tokens):
        L = tokens.size(1)
        x = self.token_emb(tokens) + self.pos_emb[:, :L]
        mask = nn.Transformer.generate_square_subsequent_mask(L).to(tokens.device)
        x = self.transformer(x, mask=mask, src_key_padding_mask=(tokens == PAD))
        x = self.norm(x)
        # feature at EOS position (last non-pad token that saw everything)
        eos_pos = (tokens == EOS).int().argmax(dim=1)  # (B,)
        return x[torch.arange(x.size(0), device=x.device), eos_pos]  # (B, hidden)


class CLIPModel(nn.Module):
    def __init__(self, hidden=768, img_size=224, patch=16, text_layers=8, text_heads=12):
        super().__init__()
        self.image_encoder = build_encoder("convnext", img_size=img_size,
                                           dim=hidden, patch=patch)
        img_dim = self.image_encoder.out_dim
        self.image_proj = nn.Linear(img_dim, hidden, bias=False)
        self.text_encoder = TextEncoder(vocab_size=VOCAB_SIZE, hidden=hidden,
                                        layers=text_layers, heads=text_heads)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / 0.07)))

    def encode_image(self, images):
        feat = self.image_proj(self.image_encoder(images))
        return F.normalize(feat, dim=-1)

    def encode_text(self, tokens):
        feat = self.text_encoder(tokens)
        return F.normalize(feat, dim=-1)

    def forward(self, images, tokens):
        return self.encode_image(images), self.encode_text(tokens)


def gather_with_grad(t, world):
    """All-gather tensor across GPUs; local copy keeps gradient, others detached."""
    if world <= 1:
        return t
    gathered = [torch.zeros_like(t) for _ in range(world)]
    dist.all_gather(gathered, t.contiguous())
    gathered[dist.get_rank()] = t
    return torch.cat(gathered, dim=0)


def clip_loss(img_feat, txt_feat, logit_scale, world=1):
    """Symmetric InfoNCE. Gather both image and text for full world*B x world*B matrix."""
    img_all = gather_with_grad(img_feat, world)  # (world*B, D)
    txt_all = gather_with_grad(txt_feat, world)  # (world*B, D)
    logits = img_all @ txt_all.T * logit_scale   # (world*B, world*B)
    N = img_all.size(0)
    labels = torch.arange(N, device=img_feat.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2


@torch.no_grad()
def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tar_dir", required=True)
    p.add_argument("--num_tars", type=int, default=81)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--text_layers", type=int, default=8)
    p.add_argument("--text_heads", type=int, default=12)
    p.add_argument("--max_len", type=int, default=77)
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
    p.add_argument("--out", default="./outputs/clip")
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
    torch.manual_seed(args.seed)

    ds = TarImageTextCLIP(args.tar_dir, args.num_tars, args.img_size, args.max_len)
    if world > 1:
        sampler = DistributedSampler(ds, shuffle=True, drop_last=True)
        train = DataLoader(ds, batch_size=args.batch, sampler=sampler,
                           num_workers=args.workers, drop_last=True,
                           persistent_workers=args.workers > 0, pin_memory=True)
    else:
        train = DataLoader(ds, batch_size=args.batch, shuffle=True,
                           num_workers=args.workers, drop_last=True,
                           persistent_workers=args.workers > 0, pin_memory=True)

    model = CLIPModel(args.hidden, args.img_size, 16, args.text_layers, args.text_heads).to(device)
    if world > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    base = model.module if isinstance(model, DDP) else model
    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[model] trainable params(M)={n_params/1e6:.1f}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, total_steps, args.warmup))

    step = 0
    for epoch in range(args.epochs):
        if world > 1:
            sampler.set_epoch(epoch)
        model.train()
        bar = tqdm(train, desc=f"e{epoch}", disable=not is_main,
                   dynamic_ncols=True, mininterval=2.0)
        for images, tokens in bar:
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp, dtype=torch.bfloat16):
                img_feat, txt_feat = model(images, tokens)
                logit_scale = base.logit_scale.exp().clamp(max=100.0)
                loss = clip_loss(img_feat, txt_feat, logit_scale, world)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            if is_main and step % args.log_every == 0:
                with torch.no_grad():
                    sim = (img_feat @ txt_feat.T).mean().item()
                bar.set_postfix(loss=f"{loss.item():.4f}", sim=f"{sim:.3f}",
                                lr=f"{sched.get_last_lr()[0]:.1e}")
            step += 1
        if is_main:
            torch.save({"model": base.state_dict(),
                        "args": {**vars(args), "arch": "clip"}},
                       out / f"epoch{epoch}.pt")
            print(f"== epoch {epoch} done, loss={loss.item():.4f} ==", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
