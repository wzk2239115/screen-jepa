import argparse
import csv
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as Fn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, random_split
from tqdm import tqdm

from dataset import TextImageDataset, load_sentences
from model import TextJEPA


def build_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--corpus_txt", default=None)
    p.add_argument("--hf_dataset", default=None)
    p.add_argument("--hf_config", default=None)
    p.add_argument("--hf_split", default="train")
    p.add_argument("--language", default=None)
    p.add_argument("--parquet_path",
                   default="/home/wzk/projects/screen-jepa/data/common_corpus_sample/common_corpus_1/subset_100_1.parquet")
    p.add_argument("--max_sentences", type=int, default=20000)
    p.add_argument("--ascii_only", type=int, default=1)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--font_size", type=int, default=72)
    p.add_argument("--mask_ratio", type=float, default=0.2)
    # model
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--mlp_dim", type=int, default=3072)
    p.add_argument("--embed_dim", type=int, default=0)
    p.add_argument("--patch_size", type=int, default=16,
                   help="16=fast but ~1.3 chars/patch (char-blind); 8 or 4 resolves glyphs")
    p.add_argument("--arch", default="vit",
                   choices=["vit", "convnext", "convvit", "windowvit",
                            "hiera", "pvt", "retina", "efficient", "mamba"])
    # optim
    p.add_argument("--batch", type=int, default=192)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # runtime
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--bf16", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="./outputs/run1")
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--objective", default="invariance", choices=["invariance", "predictive"])
    p.add_argument("--pred_depth", type=int, default=4)
    p.add_argument("--ema_tau", type=float, default=0.996)
    p.add_argument("--target_mode", default="ema", choices=["ema", "stopgrad"])
    p.add_argument("--grid", type=int, default=14)
    p.add_argument("--bg_augment", type=int, default=0,
                   help="vary background color between views (force text-focus)")
    p.add_argument("--font_augment", type=int, default=0,
                   help="vary font between views (prevent shape memorization)")
    p.add_argument("--font_pool", default=None,
                   help="comma-separated .ttf paths for font augmentation")
    p.add_argument("--geom_strength", type=int, default=0,
                   help="0=off, 1=light, 2=medium geometry augmentation (scale/rotation/shear)")
    return p.parse_args()


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


@torch.no_grad()
def evaluate(base, loader, device, amp, is_pred=False):
    base.eval()
    loss_sum, n = 0.0, 0
    Fs, Ms = [], []
    for batch in loader:
        if is_pred:
            full, masked, cell_mask = batch
            cell_mask = cell_mask.to(device, non_blocking=True)
        else:
            full, masked = batch
        full = full.to(device, non_blocking=True)
        masked = masked.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp, dtype=torch.bfloat16):
            if is_pred:
                loss, _ = base(full, masked, cell_mask)
                loss_sum += float(loss) * full.size(0)
            zf = base.encode(full).float()
            zm = base.encode(masked).float()
        Fs.append(zf.cpu()); Ms.append(zm.cpu())
        n += full.size(0)
    F = torch.cat(Fs); M = torch.cat(Ms)
    same = Fn.cosine_similarity(F, M, dim=-1).mean().item()
    diff = Fn.cosine_similarity(F, F[torch.randperm(F.size(0))], dim=-1).mean().item()
    std = F.std(dim=0).mean().item()
    dead = (F.std(dim=0) < 1e-2).float().mean().item()
    base.train()
    metric = (loss_sum / n) if is_pred else (float(((F - M) ** 2).mean()))
    return metric, std, dead, same, diff


def main():
    args = build_args()
    amp = bool(args.bf16)
    embed_dim = args.embed_dim or args.hidden
    rank, local_rank, world = setup_ddp()
    is_main = rank == 0
    device = f"cuda:{local_rank}"

    if is_main:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    if is_main:
        used_files = []
        sents = load_sentences(args.corpus_txt, args.parquet_path, args.max_sentences,
                               ascii_only=bool(args.ascii_only), hf_dataset=args.hf_dataset,
                               hf_config=args.hf_config, hf_split=args.hf_split,
                               language=args.language, used=used_files)
        if len(sents) < 100:
            raise RuntimeError(f"only {len(sents)} sentences loaded; check corpus")
        print(f"[data] sentences: {len(sents)}", flush=True)
        if used_files:
            (out / "datasource.txt").write_text("\n".join(used_files))
            print(f"[data] pinned parquet files -> {out/'datasource.txt'}", flush=True)
    else:
        sents = None

    if world > 1:
        obj = [sents]
        dist.broadcast_object_list(obj, src=0)
        sents = obj[0]

    n_val = max(64, int(len(sents) * args.val_frac))
    n_train = len(sents) - n_val
    g = torch.Generator().manual_seed(args.seed)
    is_pred = args.objective == "predictive"
    full_ds = TextImageDataset(sents, args.img_size, args.font_size, args.mask_ratio,
                               grid=args.grid, return_cell_mask=is_pred,
                               bg_augment=bool(args.bg_augment),
                               font_augment=bool(args.font_augment),
                               font_pool=args.font_pool,
                               geom_strength=args.geom_strength)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

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

    if is_pred:
        from pred_model import PredictiveJEPA
        model = PredictiveJEPA(arch=args.arch, img_size=args.img_size, hidden=args.hidden,
                               layers=args.layers, heads=args.heads, patch=args.patch_size,
                               pred_depth=args.pred_depth, ema_tau=args.ema_tau,
                               target_mode=args.target_mode).to(device)
        ddp_kw = dict(device_ids=[local_rank], find_unused_parameters=True)
    else:
        model = TextJEPA(args.hidden, args.layers, args.heads, args.mlp_dim,
                         patch=args.patch_size, img_size=args.img_size, embed_dim=embed_dim,
                         arch=args.arch).to(device)
        ddp_kw = dict(device_ids=[local_rank], static_graph=True)
    if world > 1:
        model = DDP(model, **ddp_kw)
    base = model.module if isinstance(model, DDP) else model
    if is_main:
        print(f"[model] objective={args.objective} arch={args.arch} "
              f"params(M): {sum(p.numel() for p in model.parameters())/1e6:.2f} "
              f"hidden={args.hidden} layers={args.layers} world={world} eff_batch={args.batch*world}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, total_steps, args.warmup))

    if is_main:
        csv_path = Path(args.out) / "log.csv"
        with csv_path.open("w", newline="") as f:
            hdr = ["epoch", "step", "loss", "pred" if is_pred else "inv", "reg",
                   "std", "dead_dim", "cos", "lr", "dt_s"]
            csv.writer(f).writerow(hdr)
        val_csv_path = Path(args.out) / "val_log.csv"
        with val_csv_path.open("w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss", "std", "dead_dim",
                                    "cos_same", "cos_diff", "margin"])

    step = 0
    for epoch in range(args.epochs):
        if world > 1:
            sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        bar = tqdm(train, desc=f"e{epoch}", disable=not is_main,
                   dynamic_ncols=True, mininterval=1.0)
        for batch in bar:
            if is_pred:
                full, masked, cell_mask = batch
                cell_mask = cell_mask.to(device, non_blocking=True)
            else:
                full, masked = batch
            full = full.to(device, non_blocking=True)
            masked = masked.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp, dtype=torch.bfloat16):
                if is_pred:
                    loss, stats = model(full, masked, cell_mask, lam=args.lam)
                else:
                    loss, stats = model(full, masked, lam=args.lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            if is_pred:
                base.update_ema()

            if is_main and step % args.log_every == 0:
                with torch.no_grad():
                    zf = base.encode(full).float()
                    zm = base.encode(masked).float()
                    cos = Fn.cosine_similarity(zf, zm, dim=-1).mean().item()
                    zstd = zf.std(dim=0).mean().item()
                    dead = (zf.std(dim=0) < 1e-2).float().mean().item()
                key = "pred" if is_pred else "inv"
                bar.set_postfix(loss=f"{loss.item():.4f}", **{key: f"{float(stats[key]):.4f}"},
                                std=f"{zstd:.3f}", dead=f"{dead:.3f}",
                                cos=f"{cos:.3f}", lr=f"{sched.get_last_lr()[0]:.1e}")
                regv = float(stats["reg"]) if "reg" in stats else 0.0
                with csv_path.open("a", newline="") as f:
                    csv.writer(f).writerow([epoch, step, f"{loss.item():.5f}",
                                            f"{float(stats[key]):.5f}",
                                            f"{regv:.5f}",
                                            f"{zstd:.5f}", f"{dead:.5f}",
                                            f"{cos:.5f}",
                                            f"{sched.get_last_lr()[0]:.3e}",
                                            f"{bar.format_dict.get('rate') or 0.0:.3f}"])
            step += 1

        if is_main:
            v_loss, v_std, v_dead, v_same, v_diff = evaluate(base, val, device, amp, is_pred)
            print(f"== epoch {epoch} val: loss={v_loss:.4f} std={v_std:.3f} dead={v_dead:.3f} "
                  f"cos_same={v_same:.3f} cos_diff={v_diff:.3f} margin={v_same-v_diff:.3f} ==", flush=True)
            with val_csv_path.open("a", newline="") as f:
                csv.writer(f).writerow([epoch, f"{v_loss:.5f}", f"{v_std:.5f}",
                                        f"{v_dead:.5f}", f"{v_same:.5f}",
                                        f"{v_diff:.5f}", f"{v_same-v_diff:.5f}"])
            if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
                torch.save({"model": base.state_dict(), "args": vars(args)},
                           Path(args.out) / f"epoch{epoch}.pt")

    if world > 1:
        dist.destroy_process_group()
    if is_main:
        print("[done]", flush=True)


if __name__ == "__main__":
    main()
