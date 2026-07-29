"""Train TC-JEPA on image-caption pairs (recap-datacomp).

Launch (8 × H800):
    torchrun --nproc_per_node=8 --rdzv-endpoint 127.0.0.1:29500 \
        train_tc_jepa.py --tar_dir /home/jovyan/h800fast/wangzekai/recap-datacomp-384-1M \
        --out outputs/tcjepa_v1
"""

import io
import json
import os
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from tc_jepa import TCJEPA


# ============================ Dataset ============================

_TAR_CACHE = {}


def _tar(path):
    if path not in _TAR_CACHE:
        _TAR_CACHE[path] = __import__("tarfile").open(path)
    return _TAR_CACHE[path]


class TarImageCaption(Dataset):
    """Stream images + raw captions from webdataset-style tars."""

    def __init__(self, tar_dir, num_tars=None, img_size=224, augment=True):
        import tarfile
        tars = sorted(str(p) for p in Path(tar_dir).glob("*.tar"))
        if num_tars:
            tars = tars[:num_tars]
        self.index = []
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
                print(f"[data] WARNING corrupt tar {tp}: {e}", flush=True)
        self.img_size = img_size

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.3, 1.0), ratio=(0.75, 1.33)),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transform = transforms.Resize((img_size, img_size))

        self.norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        print(f"[data] indexed {len(self.index)} pairs from {good}/{len(tars)} tars", flush=True)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        import random as _r
        for _ in range(5):
            try:
                tp, name = self.index[_r.randint(0, len(self.index) - 1)]
                tf = _tar(tp)
                img = Image.open(io.BytesIO(tf.extractfile(name).read())).convert("RGB")
                cap = json.loads(tf.extractfile(name.replace(".jpg", ".json")).read())["caption"]
                img = self.transform(img)
                t = transforms.functional.to_tensor(img)
                t = self.norm(t)
                return t, cap
            except Exception:
                continue
        return torch.zeros(3, self.img_size, self.img_size), ""


def make_collate(tokenizer, max_length=77):
    def collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        caps = [b[1] if b[1] else "a photo" for b in batch]
        enc = tokenizer(
            caps, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        return imgs, enc["input_ids"], enc["attention_mask"]
    return collate


# ============================ Train ============================

def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world = dist.get_world_size()
    else:
        rank = local_rank = 0
        world = 1
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world


def lr_lambda(step, total, warmup):
    w = max(1, int(total * warmup))
    if step < w:
        return step / w
    prog = (step - w) / max(1, total - w)
    return 0.5 * (1 + math.cos(math.pi * prog))


def build_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tar_dir", required=True)
    p.add_argument("--num_tars", type=int, default=81)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--patch_size", type=int, default=16)

    # encoder
    p.add_argument("--encoder_dim", type=int, default=768)
    p.add_argument("--encoder_depth", type=int, default=12)
    p.add_argument("--encoder_heads", type=int, default=12)

    # predictor
    p.add_argument("--pred_dim", type=int, default=384)
    p.add_argument("--pred_depth", type=int, default=6)
    p.add_argument("--pred_heads", type=int, default=12)

    # text
    p.add_argument("--t5_model", type=str, default="t5-small")
    p.add_argument("--max_caption_len", type=int, default=77)

    # loss coefficients
    p.add_argument("--lam_sparse", type=float, default=0.1)
    p.add_argument("--lam_consistency", type=float, default=0.5)

    # masking
    p.add_argument("--num_target_blocks", type=int, default=4)
    p.add_argument("--target_scale_min", type=float, default=0.10)
    p.add_argument("--target_scale_max", type=float, default=0.25)

    # optimization
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.04)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ema_tau", type=float, default=0.996)

    # augmentation
    p.add_argument("--augment", type=int, default=1)

    # logging
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=10)

    p.add_argument("--bf16", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="./outputs/tcjepa")
    return p.parse_args()


def main():
    args = build_args()
    rank, local_rank, world = setup_ddp()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(args.seed + rank)

    out = Path(args.out)
    if is_main:
        out.mkdir(parents=True, exist_ok=True)

    # ---- model ----
    if is_main:
        print("[init] building model + loading T5 (first run downloads ~250MB)...", flush=True)
    model = TCJEPA(
        img_size=args.img_size,
        patch_size=args.patch_size,
        encoder_dim=args.encoder_dim,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        pred_dim=args.pred_dim,
        pred_depth=args.pred_depth,
        pred_heads=args.pred_heads,
        t5_model=args.t5_model,
        lam_sparse=args.lam_sparse,
        lam_consistency=args.lam_consistency,
        ema_tau=args.ema_tau,
        target_scale=(args.target_scale_min, args.target_scale_max),
        num_target_blocks=args.num_target_blocks,
    ).to(device)

    # ---- DDP ----
    if world > 1:
        find = {"find_unused_parameters": False}
        model_ddp = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], **find,
        )
    else:
        model_ddp = model
    base = model_ddp.module if world > 1 else model

    # ---- data ----
    if is_main:
        print("[init] indexing dataset tars...", flush=True)
    ds = TarImageCaption(
        args.tar_dir, args.num_tars, args.img_size, augment=bool(args.augment),
    )
    sampler = DistributedSampler(ds) if world > 1 else None
    collate = make_collate(model.text_encoder.tokenizer, args.max_caption_len)
    train = DataLoader(
        ds, batch_size=args.batch, sampler=sampler, shuffle=sampler is None,
        num_workers=args.workers, pin_memory=True, drop_last=True, collate_fn=collate,
    )

    # ---- optimizer (only trainable params: encoder + predictor) ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, total_steps, args.warmup),
    )

    if is_main:
        n_train = sum(p.numel() for p in trainable)
        print(f"[model] trainable params: {n_train/1e6:.1f}M", flush=True)
        print(f"[train] {len(train)} steps/epoch × {args.epochs} epochs = {total_steps} steps", flush=True)
        print("[init] starting training...", flush=True)

    # ---- tensorboard ----
    writer = None
    if is_main:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(out / "tb")

    # ---- train loop ----
    amp = bool(args.bf16)
    step = 0
    for epoch in range(args.epochs):
        if world > 1:
            sampler.set_epoch(epoch)
        model.train()
        bar = tqdm(train, desc=f"e{epoch}", dynamic_ncols=True, mininterval=2.0,
                   disable=not is_main)
        for imgs, ids, mask in bar:
            imgs = imgs.to(device, non_blocking=True)
            ids = ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp, dtype=torch.bfloat16):
                loss, stats = model(imgs, ids, mask)

            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            if torch.isnan(gn) or torch.isinf(gn):
                if is_main:
                    print(f"[WARN] NaN/Inf grad at step {step}, skipping", flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            sched.step()
            base.update_ema(epoch, args.epochs)

            if is_main and step % args.log_every == 0:
                lr = sched.get_last_lr()[0]
                bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    l2=f"{float(stats['l2']):.3f}",
                    cos=f"{float(stats['cos_pt']):.3f}",
                    sp=f"{float(stats['sparse']):.3f}",
                    lr=f"{lr:.1e}",
                )
                if writer:
                    writer.add_scalar("train/loss", loss.item(), step)
                    writer.add_scalar("train/l2", float(stats["l2"]), step)
                    writer.add_scalar("train/cos_pt", float(stats["cos_pt"]), step)
                    writer.add_scalar("train/sparse", float(stats["sparse"]), step)
                    writer.add_scalar("train/consistency", float(stats["consistency"]), step)
                    writer.add_scalar("train/lr", lr, step)
                    writer.add_scalar("train/grad_norm", float(gn), step)
            step += 1

        # ---- end of epoch ----
        do_save = (epoch % args.save_every == 0) or (epoch == args.epochs - 1)
        if do_save and is_main:
            torch.save(
                {"model": base.state_dict(), "args": vars(args), "objective": "tc_jepa"},
                out / f"epoch{epoch}.pt",
            )
            print(f"   [saved checkpoint epoch {epoch}]", flush=True)

        # ---- eval: feature collapse check ----
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                imgs_eval = imgs[:16]
                z = base.encoder(imgs_eval)
                feat_std = z.std(dim=1).mean().item()
                # feature rank (effective dimensionality)
                z_flat = z.reshape(-1, z.size(-1))
                if z_flat.size(0) > 1:
                    cov = torch.cov(z_flat.T)
                    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
                    eff_rank = (eigvals.sum() ** 2 / (eigvals ** 2).sum()).item()
                else:
                    eff_rank = 0.0
            if is_main:
                print(
                    f"== epoch {epoch} eval: feat_std={feat_std:.4f} "
                    f"eff_rank={eff_rank:.1f}/{z.size(-1)} ==",
                    flush=True,
                )
                if writer:
                    writer.add_scalar("eval/feat_std", feat_std, epoch)
                    writer.add_scalar("eval/eff_rank", eff_rank, epoch)
            model.train()

    if is_main and writer:
        writer.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
