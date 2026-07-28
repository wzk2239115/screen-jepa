"""Layer-wise probe: where does "shape capture ability" live in the network?

Extracts features at each backbone stage, each stage2 block, and enhancer ticks,
then evaluates word-photo retrieval quality for each layer. This reveals the
depth distribution of semantic understanding.

Usage:
  python probe_layers.py --ckpt outputs/jepa_decay10/epoch10.pt \
    --tar_dir /path/to/recap-datacomp-384-1M
"""
import argparse
import io
import json
import math
import random
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from render import DEFAULT_FONT
from train_crossmodal_jepa import build_composite, boxes_to_cell_masks, region_mask

STOP = set("""the a an of in on at to for and or but is are was were be been being
with without from by as this that these those it its their his her my your our we
they he she i you very into over under above below up down out off again further
then once here there all any both each few more most other some such no nor not
only own same so than too can will just don should now which who whom whose what
where when why how also has have had do does did doing would could may might must
shall about against between through during before after along across behind beyond
near outside past round since till until within two three four five six seven
eight nine ten first second third next last new old big small long great little own
look looking looked feel feeling felt get getting got go going went one two
image photo picture close up view background foreground front back side top bottom
left right center middle wall floor ground sky""".split())


def clean_word(w):
    return w.lower().strip(".,!?;:'\"()[]{}")


def load_samples(tar_dir, n, max_tars=10):
    tars = sorted(Path(tar_dir).glob("*.tar"))[:max_tars]
    out = []
    for tp in tars:
        try:
            tf = tarfile.open(tp)
        except Exception:
            continue
        for m in tf.getmembers():
            if m.name.endswith(".jpg"):
                try:
                    jf = tf.extractfile(m.name.replace(".jpg", ".json"))
                    cap = json.loads(jf.read())["caption"]
                    img_bytes = tf.extractfile(m.name).read()
                    out.append((cap, img_bytes))
                    if len(out) >= n:
                        return out
                except Exception:
                    continue
        tf.close()
    return out


def pool_regions(feat, grid):
    """Pool text (top half) and photo (bottom half) from feature map.
    
    feat: (B, N, C) where N = grid*grid, OR (B, C, H, W)
    Returns: text_feat (B, C), photo_feat (B, C)
    """
    if feat.dim() == 4:
        B, C, H, W = feat.shape
        feat = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
        g = H
    else:
        g = int(math.sqrt(feat.shape[1]))
    
    mid = g // 2
    text_cells = list(range(mid * g))       # top half rows
    photo_cells = list(range(mid * g, g * g))  # bottom half rows
    
    text_feat = feat[:, text_cells].mean(dim=1)    # (B, C)
    photo_feat = feat[:, photo_cells].mean(dim=1)   # (B, C)
    return text_feat, photo_feat


def compute_alignment(text_feats, photo_feats):
    """Compute text-photo alignment quality.
    
    Returns: (matched_cos, unmatched_cos, gap, top5)
    """
    text_n = F.normalize(text_feats, dim=-1)
    photo_n = F.normalize(photo_feats, dim=-1)
    
    sim = text_n @ photo_n.T  # (B, B)
    B = sim.shape[0]
    
    diag = sim.diag().mean().item()
    off_mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
    off = sim[off_mask].mean().item()
    
    # top-5 retrieval: for each text, how many of top-5 photos include the match?
    top5_hits = 0
    for i in range(B):
        top5_idx = sim[i].topk(5).indices.tolist()
        if i in top5_idx:
            top5_hits += 1
    top5 = top5_hits / B
    
    return diag, off, diag - off, top5


def probe_layer(features_list, grid):
    """Evaluate alignment for a list of feature maps.
    features_list: list of (B, N, C) tensors, one per batch
    Returns aggregated alignment metrics.
    """
    all_text = []
    all_photo = []
    for feat in features_list:
        t, p = pool_regions(feat, grid)
        all_text.append(t)
        all_photo.append(p)
    
    text = torch.cat(all_text, dim=0)
    photo = torch.cat(all_photo, dim=0)
    
    return compute_alignment(text, photo)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tar_dir", required=True)
    p.add_argument("--n_samples", type=int, default=300)
    args = p.parse_args()

    device = "cuda"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    margs = ck["args"]
    img_size = margs["img_size"]
    grid = img_size // margs["patch_size"]
    patch = margs["patch_size"]

    from train_ctm_enc_jepa import CTMEncoderJepa
    model = CTMEncoderJepa(
        "convnext", img_size, margs["hidden"], margs["layers"],
        margs["heads"], margs["patch_size"], margs.get("pred_depth", 4),
        margs.get("ema_tau", 0.996), margs.get("ctm_iters", 50),
        margs.get("ctm_memory", 4), margs.get("ctm_thoughts", 8),
        margs.get("bptt_window", 15)).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    print(f"[probe] loaded {args.ckpt}", flush=True)

    samples = load_samples(args.tar_dir, args.n_samples)
    random.seed(0)
    random.shuffle(samples)
    print(f"[probe] {len(samples)} samples", flush=True)

    backbone = model.backbone
    enhancer = model.enhancer

    # === collect features at each layer ===
    layer_feats = defaultdict(list)

    with torch.no_grad():
        for cap, img_bytes in samples:
            img = Image.open(io.BytesIO(img_bytes))
            composite, _ = build_composite(cap, img)
            t = torch.from_numpy(composite).float().permute(2, 0, 1) / 255.0
            t = ((t - 0.5) / 0.5).unsqueeze(0).to(device)

            # --- backbone forward with manual hooks ---
            x = t
            for i in range(4):
                x = backbone.down_layers[i](x)
                x = backbone.stages[i](x)
                layer_feats[f"stage{i}"].append(x.cpu())
                if i == 2:
                    # capture individual stage2 blocks
                    pass
            # stage2 individual blocks (re-run to capture per-block)
            x2 = backbone.down_layers[2](backbone.down_layers[1](
                 backbone.down_layers[0](t)))
            for j, blk in enumerate(backbone.stages[2]):
                x2 = blk(x2)
                layer_feats[f"s2_blk{j}"].append(x2.cpu())
            ctx = x2.flatten(2).transpose(1, 2)  # (1, 196, 768)

            # --- enhancer forward with tick captures ---
            B, N, D = ctx.shape
            K, M = enhancer.num_thoughts, enhancer.memory_length
            thoughts = enhancer.thought_init.expand(B, K, D)
            trace = thoughts.unsqueeze(-1).expand(B, K, D, M).contiguous()
            tick_targets = [1, 3, 5, 10, 20, 35, 50]

            def broadcast(thoughts, ctx):
                q = enhancer.out_q_proj(ctx)
                kv = enhancer.out_kv_proj(thoughts)
                k, v = kv.chunk(2, dim=-1)
                hd = D // enhancer.num_heads
                qh = q.reshape(B, N, enhancer.num_heads, hd).transpose(1, 2)
                kh = k.reshape(B, K, enhancer.num_heads, hd).transpose(1, 2)
                vh = v.reshape(B, K, enhancer.num_heads, hd).transpose(1, 2)
                out = F.scaled_dot_product_attention(qh, kh, vh)
                out = out.transpose(1, 2).reshape(B, N, D)
                return enhancer.out_norm(ctx + enhancer.out_attn_proj(out))

            for tick in range(1, 51):
                thoughts, trace = enhancer._tick(thoughts, ctx, trace)
                if tick in tick_targets:
                    bc = broadcast(thoughts, ctx)
                    layer_feats[f"enh_t{tick}"].append(bc.cpu())

    # === evaluate each layer ===
    layer_order = ([f"stage{i}" for i in range(4)] +
                   [f"s2_blk{j}" for j in range(6)] +
                   [f"enh_t{t}" for t in tick_targets])
    print(f"\n{'layer':<16s} {'matched':>8s} {'unmatch':>8s} {'gap':>8s} {'top-5':>8s}")
    print("-" * 52)

    results = []
    for name in layer_order:
        if name not in layer_feats:
            continue
        feats = layer_feats[name]
        if feats[0].dim() == 4:
            g = feats[0].shape[-1]  # H or W
        else:
            g = int(math.sqrt(feats[0].shape[1]))
        matched, unmatch, gap, top5 = probe_layer(feats, g)
        results.append((name, matched, unmatch, gap, top5))
        print(f"{name:<16s} {matched:8.4f} {unmatch:8.4f} {gap:8.4f} {top5:8.3f}")

    # === summary ===
    print("\n=== summary ===")
    best_gap = max(results, key=lambda r: r[3])
    best_top5 = max(results, key=lambda r: r[4])
    print(f"best alignment gap: {best_gap[0]} (gap={best_gap[3]:.4f})")
    print(f"best top-5:         {best_top5[0]} (top5={best_top5[4]:.3f})")

    # backbone vs enhancer comparison
    bb_results = [r for r in results if not r[0].startswith("enh")]
    enh_results = [r for r in results if r[0].startswith("enh")]
    if bb_results and enh_results:
        bb_best = max(bb_results, key=lambda r: r[3])
        enh_best = max(enh_results, key=lambda r: r[3])
        print(f"\nbackbone best: {bb_best[0]} gap={bb_best[3]:.4f} top5={bb_best[4]:.3f}")
        print(f"enhancer best: {enh_best[0]} gap={enh_best[3]:.4f} top5={enh_best[4]:.3f}")
        delta = enh_best[3] - bb_best[3]
        print(f"enhancer adds: {delta:+.4f} gap ({delta/max(bb_best[3],1e-6)*100:+.1f}%)")


if __name__ == "__main__":
    main()
