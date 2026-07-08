import argparse
import csv
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from dataset import TextImageDataset, load_sentences
from model import TextJEPA


def build_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--corpus_txt", default=None)
    p.add_argument("--hf_dataset", default=None,
                   help="e.g. PleIAs/common_corpus (streaming, uses HF cache)")
    p.add_argument("--hf_config", default=None)
    p.add_argument("--hf_split", default="train")
    p.add_argument("--language", default=None,
                   help="filter HF dataset 'language' column, e.g. English")
    p.add_argument("--parquet_path",
                   default="/home/wzk/projects/screen-jepa/data/common_corpus_sample/common_corpus_1/subset_100_1.parquet")
    p.add_argument("--max_sentences", type=int, default=20000)
    p.add_argument("--ascii_only", type=int, default=1)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--font_size", type=int, default=16)
    p.add_argument("--mask_ratio", type=float, default=0.15)
    # model
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--mlp_dim", type=int, default=2048)
    p.add_argument("--embed_dim", type=int, default=0)
    # optim
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # runtime
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--bf16", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="./outputs/run1")
    p.add_argument("--val_frac", type=float, default=0.05)
    return p.parse_args()


def lr_lambda(step, total, warmup):
    w = max(1, int(total * warmup))
    if step < w:
        return step / w
    prog = (step - w) / max(1, total - w)
    return 0.5 * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    inv_sum, n = 0.0, 0
    stds, dead = [], []
    for full, masked in loader:
        full = full.to(device, non_blocking=True)
        masked = masked.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp, dtype=torch.bfloat16):
            z_full, z_masked = model(full, masked)
        inv_sum += float(((z_full.float() - z_masked.float()) ** 2).mean()) * full.size(0)
        std = z_full.float().std(dim=0)
        stds.append(std)
        dead.append((std < 1e-2).float())
        n += full.size(0)
    std = torch.cat(stds).mean(0)
    return inv_sum / n, float(std.mean()), float(torch.cat(dead).mean(0))


def main():
    args = build_args()
    amp = bool(args.bf16)
    embed_dim = args.embed_dim or args.hidden
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = "cuda"

    sents = load_sentences(args.corpus_txt, args.parquet_path,
                           args.max_sentences, ascii_only=bool(args.ascii_only),
                           hf_dataset=args.hf_dataset, hf_config=args.hf_config,
                           hf_split=args.hf_split, language=args.language)
    if len(sents) < 100:
        raise RuntimeError(f"only {len(sents)} sentences loaded; check corpus")
    print(f"[data] sentences: {len(sents)}")

    n_val = max(64, int(len(sents) * args.val_frac))
    n_train = len(sents) - n_val
    g = torch.Generator().manual_seed(args.seed)
    full_ds = TextImageDataset(sents, args.img_size, args.font_size, args.mask_ratio)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    train = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, drop_last=True,
                       persistent_workers=args.workers > 0, pin_memory=True)
    val = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = TextJEPA(args.hidden, args.layers, args.heads, args.mlp_dim,
                     img_size=args.img_size, embed_dim=embed_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params(M): {n_params/1e6:.2f}  hidden={args.hidden} layers={args.layers}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, total_steps, args.warmup))

    csv_path = out / "log.csv"
    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "step", "loss", "inv", "reg", "z_std",
                                "dead_dim", "cos", "lr", "dt_s"])

    scaler = None
    step = 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for full, masked in train:
            full = full.to(device, non_blocking=True)
            masked = masked.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp, dtype=torch.bfloat16):
                loss, stats = model.loss(full, masked, lam=args.lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()

            if step % args.log_every == 0:
                with torch.no_grad():
                    zf = model.encode(full).float()
                    zm = model.encode(masked).float()
                    cos = torch.nn.functional.cosine_similarity(zf, zm, dim=-1).mean()
                zstd = float(zf.std(dim=0).mean())
                dead = float((zf.std(dim=0) < 1e-2).float().mean())
                dt = (time.time() - t0) / max(1, step % args.log_every or args.log_every)
                print(f"e{epoch} s{step:>6} loss={loss.item():.4f} "
                      f"inv={float(stats['inv']):.4f} reg={float(stats['reg']):.4f} "
                      f"std={zstd:.3f} dead={dead:.3f} cos={float(cos):.3f} "
                      f"lr={sched.get_last_lr()[0]:.2e} {dt:.2f}s/it")
                with csv_path.open("a", newline="") as f:
                    csv.writer(f).writerow([epoch, step, f"{loss.item():.5f}",
                                            f"{float(stats['inv']):.5f}",
                                            f"{float(stats['reg']):.5f}",
                                            f"{zstd:.5f}", f"{dead:.5f}",
                                            f"{float(cos):.5f}",
                                            f"{sched.get_last_lr()[0]:.3e}", f"{dt:.3f}"])
            step += 1

        v_inv, v_std, v_dead = evaluate(model, val, device, amp)
        print(f"== epoch {epoch} val: inv={v_inv:.4f} std={v_std:.3f} dead={v_dead:.3f} ==")
        torch.save({"model": model.state_dict(), "args": vars(args)},
                   out / f"epoch{epoch}.pt")

    print("[done]")


if __name__ == "__main__":
    main()
