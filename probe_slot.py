"""Slot JEPA zero-shot probe — mirrors probe_zeroshot.py but extracts word
and photo features via slot attention (not cell mean pooling).

This is necessary because probe_zeroshot uses model.encoder directly and
would miss the slot attention's contribution entirely.
"""
import argparse
import io
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train_slot_jepa import SlotJEPA
from train_crossmodal_jepa import build_composite, boxes_to_cell_masks, region_mask
from probe_zeroshot import load_samples, clean_word, STOP


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tar_dir", required=True)
    p.add_argument("--n_proto", type=int, default=500)
    p.add_argument("--n_test", type=int, default=300)
    p.add_argument("--min_count", type=int, default=5)
    p.add_argument("--blank_text", type=int, default=1)
    args = p.parse_args()

    device = "cuda"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    margs = ck["args"]
    img_size = margs["img_size"]
    grid = img_size // margs["patch_size"]
    patch = margs["patch_size"]
    print(f"[slot-zs] ckpt={args.ckpt} grid={grid} slots={margs.get('num_slots', 16)}", flush=True)

    model = SlotJEPA("convnext", img_size, margs["hidden"], margs["layers"],
                     margs["heads"], margs["patch_size"], margs["pred_depth"],
                     margs["ema_tau"], margs.get("num_slots", 16),
                     margs.get("slot_iters", 3)).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    pc = region_mask(grid, list(range(grid // 2, grid))).to(device)

    samples = load_samples(args.tar_dir, args.n_proto + args.n_test)
    random.seed(0)
    random.shuffle(samples)
    proto_samples = samples[:args.n_proto]
    test_samples = samples[args.n_proto:args.n_proto + args.n_test]
    print(f"[slot-zs] {len(proto_samples)} proto, {len(test_samples)} test", flush=True)

    def to_t(comp):
        t = torch.from_numpy(comp).float().permute(2, 0, 1) / 255.0
        return ((t - 0.5) / 0.5).unsqueeze(0).to(device)

    # === 1. build word prototypes via slot attention ===
    word_feats = {}
    with torch.no_grad():
        for cap, img_bytes in proto_samples:
            img = Image.open(io.BytesIO(img_bytes))
            composite, boxes = build_composite(cap, img)
            ctx = model.encoder(to_t(composite), return_map=True).float()
            slots, attn = model.slot_attn(ctx, return_attn=True)
            # slots: (1, K, D), attn: (1, N, K)
            raw_words = cap.split()
            if not boxes or len(boxes) != len(raw_words):
                continue
            masks = boxes_to_cell_masks(boxes, grid, patch).float().to(device)  # (nw, N)
            word_slot_w = masks @ attn[0]  # (nw, K)
            word_slot_w = word_slot_w / word_slot_w.sum(-1, keepdim=True).clamp(min=1e-6)
            wf = F.normalize(word_slot_w @ slots[0], dim=-1)  # (nw, D)
            for w, feat in zip([clean_word(x) for x in raw_words], wf):
                if w and w not in STOP and len(w) > 2:
                    word_feats.setdefault(w, []).append(feat)

    protos = {w: F.normalize(torch.stack(v).mean(0), dim=0)
              for w, v in word_feats.items() if len(v) >= args.min_count}
    word_list = sorted(protos.keys())
    Wmat = torch.stack([protos[w] for w in word_list])
    print(f"[slot-zs] {len(word_list)} word prototypes", flush=True)
    inter = (Wmat @ Wmat.T)
    idx = torch.triu_indices(len(word_list), len(word_list), offset=1)
    inter_vals = inter[idx[0], idx[1]]
    print(f"[slot-zs] word inter-cos: mean={inter_vals.mean():.4f} std={inter_vals.std():.4f}", flush=True)

    # === 2. zero-shot image->word retrieval ===
    print(f"\n=== zero-shot image->word retrieval (slot features) ===", flush=True)
    top1, top5, top10, total = 0, 0, 0, 0
    mrr_sum = 0.0
    with torch.no_grad():
        for cap, img_bytes in test_samples:
            img = Image.open(io.BytesIO(img_bytes))
            if args.blank_text:
                composite, _ = build_composite("", img)
            else:
                composite, _ = build_composite(cap, img)
            ctx = model.encoder(to_t(composite), return_map=True).float()
            slots, attn = model.slot_attn(ctx, return_attn=True)
            photo_attn = attn[0, pc, :].mean(0)  # (K,)
            photo_attn = photo_attn / photo_attn.sum().clamp(min=1e-6)
            photo_f = F.normalize(photo_attn @ slots[0], dim=-1)

            sims = Wmat @ photo_f
            order = sims.argsort(descending=True)
            ranked = [word_list[i] for i in order]

            gt_words = set(clean_word(w) for w in cap.split()) - STOP
            gt_words = {w for w in gt_words if w in protos}
            if not gt_words:
                continue
            total += 1
            if ranked[0] in gt_words:
                top1 += 1
            if any(w in gt_words for w in ranked[:5]):
                top5 += 1
            if any(w in gt_words for w in ranked[:10]):
                top10 += 1
            for rank, w in enumerate(ranked):
                if w in gt_words:
                    mrr_sum += 1.0 / (rank + 1)
                    break

    n = max(total, 1)
    vocab = len(word_list)
    print(f"test images: {total}  vocab: {vocab}", flush=True)
    print(f"top-1:  {top1}/{total} = {top1/n:.3f}   (random={1/vocab:.4f})", flush=True)
    print(f"top-5:  {top5}/{total} = {top5/n:.3f}   (random={5/vocab:.4f})", flush=True)
    print(f"top-10: {top10}/{total} = {top10/n:.3f}   (random={10/vocab:.4f})", flush=True)
    print(f"MRR:    {mrr_sum/n:.3f}", flush=True)


if __name__ == "__main__":
    main()
