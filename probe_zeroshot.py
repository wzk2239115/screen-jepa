"""Training-distribution zero-shot probe.

Builds word prototypes from REAL composites (caption+photo, training distribution),
NOT from the OOD probe (single word + white bottom). Then does zero-shot
image->word retrieval on held-out images.

This avoids the probe-distribution collapse that invalidated probe_nouns.py
(where white-bottom composites pushed all word features to cos~0.97).
"""
import argparse
import io
import json
import random
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train_crossmodal_jepa import (CrossModalJEPA, build_composite,
                                   boxes_to_cell_masks, region_mask)

STOP = set("""the a an of in on at to for and or but is are was were be been being
with without from by as this that these those it its their his her my your our we
they he she i you very into over under above below up down out off again further
then once here there all any both each few more most other some such no nor not
only own same so than too can will just don should now which who whom whose what
where when why how also has have had do does did doing would could may might must
shall about against between through during before after above below up down out
off over under again further then once here there along across behind beyond during
near outside past round since till until within without two three four five six seven
eight nine ten first second third next last new old big small long great little own
old new look looking looked feel feeling felt get getting got go going went one two
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tar_dir", required=True)
    p.add_argument("--n_proto", type=int, default=500)
    p.add_argument("--n_test", type=int, default=300)
    p.add_argument("--min_count", type=int, default=5, help="min instances to build a word prototype")
    args = p.parse_args()

    device = "cuda"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    margs = ck["args"]
    img_size = margs["img_size"]
    grid = img_size // margs["patch_size"]
    patch = margs["patch_size"]
    print(f"[zs] ckpt={args.ckpt}  grid={grid}", flush=True)

    model = CrossModalJEPA("convnext", img_size, margs["hidden"], margs["layers"],
                           margs["heads"], margs["patch_size"], margs["pred_depth"],
                           margs["ema_tau"]).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    tc = region_mask(grid, list(range(grid // 2))).to(device)
    pc = region_mask(grid, list(range(grid // 2, grid))).to(device)

    samples = load_samples(args.tar_dir, args.n_proto + args.n_test)
    random.seed(0)
    random.shuffle(samples)
    proto_samples = samples[:args.n_proto]
    test_samples = samples[args.n_proto:args.n_proto + args.n_test]
    print(f"[zs] {len(proto_samples)} proto samples, {len(test_samples)} test samples", flush=True)

    # === 1. build word prototypes from training-distribution composites ===
    word_feats = {}
    with torch.no_grad():
        for cap, img_bytes in proto_samples:
            img = Image.open(io.BytesIO(img_bytes))
            composite, boxes = build_composite(cap, img)
            t = torch.from_numpy(composite).float().permute(2, 0, 1) / 255.0
            t = ((t - 0.5) / 0.5).unsqueeze(0).to(device)
            f = model.encoder(t, return_map=True).float()  # (1, N, D)

            raw_words = cap.split()
            clean_words = [clean_word(w) for w in raw_words]
            if not boxes or len(boxes) != len(raw_words):
                continue
            masks = boxes_to_cell_masks(boxes, grid, patch).float().to(device)  # (nw, N)
            wf = torch.einsum("wn,nd->wd", masks, f[0]) / masks.sum(1, keepdim=True)
            wf = F.normalize(wf, dim=-1)
            for w, feat in zip(clean_words, wf):
                if w and w not in STOP and len(w) > 2:
                    word_feats.setdefault(w, []).append(feat)

    protos = {w: F.normalize(torch.stack(v).mean(0), dim=0)
              for w, v in word_feats.items() if len(v) >= args.min_count}
    word_list = sorted(protos.keys())
    Wmat = torch.stack([protos[w] for w in word_list])
    print(f"[zs] {len(word_list)} word prototypes (>= {args.min_count} instances)", flush=True)
    print(f"[zs] vocab sample: {word_list[:30]}", flush=True)

    # check prototype quality: intra vs inter
    inter = (Wmat @ Wmat.T)
    inter_vals = inter[torch.triu_indices(len(word_list), len(word_list), offset=1).unbind()]
    print(f"[zs] proto inter-word cos: mean={inter_vals.mean():.4f} std={inter_vals.std():.4f}", flush=True)

    # === 2. zero-shot retrieval on held-out images ===
    print(f"\n=== zero-shot image->word retrieval ===", flush=True)
    top1, top5, top10, total = 0, 0, 0, 0
    mrr_sum = 0.0
    for cap, img_bytes in test_samples:
        img = Image.open(io.BytesIO(img_bytes))
        composite, _ = build_composite(cap, img)
        t = torch.from_numpy(composite).float().permute(2, 0, 1) / 255.0
        t = ((t - 0.5) / 0.5).unsqueeze(0).to(device)
        with torch.no_grad():
            f = model.encoder(t, return_map=True).float()
        photo_f = F.normalize(f[0, pc].mean(0), dim=0)

        sims = Wmat @ photo_f  # (num_words,)
        order = sims.argsort(descending=True)
        ranked = [word_list[i] for i in order]

        gt_words = set(clean_word(w) for w in cap.split()) - STOP
        gt_words = {w for w in gt_words if w in protos}  # only words we have prototypes for
        if not gt_words:
            continue
        total += 1

        if ranked[0] in gt_words:
            top1 += 1
        if any(w in gt_words for w in ranked[:5]):
            top5 += 1
        if any(w in gt_words for w in ranked[:10]):
            top10 += 1
        # MRR
        for rank, w in enumerate(ranked):
            if w in gt_words:
                mrr_sum += 1.0 / (rank + 1)
                break

    n = max(total, 1)
    vocab = len(word_list)
    print(f"test images: {total}", flush=True)
    print(f"vocab size:  {vocab}", flush=True)
    print(f"top-1:  {top1}/{total} = {top1/n:.3f}   (random={1/vocab:.4f})", flush=True)
    print(f"top-5:  {top5}/{total} = {top5/n:.3f}   (random={5/vocab:.4f})", flush=True)
    print(f"top-10: {top10}/{total} = {top10/n:.3f}   (random={10/vocab:.4f})", flush=True)
    print(f"MRR:    {mrr_sum/n:.3f}   (random~{1/vocab:.4f})", flush=True)

    # show a few qualitative examples
    print("\n=== qualitative examples ===", flush=True)
    shown = 0
    for cap, img_bytes in test_samples[:50]:
        img = Image.open(io.BytesIO(img_bytes))
        composite, _ = build_composite(cap, img)
        t = torch.from_numpy(composite).float().permute(2, 0, 1) / 255.0
        t = ((t - 0.5) / 0.5).unsqueeze(0).to(device)
        with torch.no_grad():
            f = model.encoder(t, return_map=True).float()
        photo_f = F.normalize(f[0, pc].mean(0), dim=0)
        sims = Wmat @ photo_f
        top5_idx = sims.topk(5).indices.tolist()
        top5_words = [word_list[i] for i in top5_idx]
        gt_words = set(clean_word(w) for w in cap.split()) - STOP
        gt_words = {w for w in gt_words if w in protos}
        hits = [w for w in top5_words if w in gt_words]
        if shown < 10 and gt_words:
            print(f"  cap: {cap[:80]}", flush=True)
            print(f"  gt:  {list(gt_words)[:8]}", flush=True)
            print(f"  top5: {top5_words}  hits={hits}", flush=True)
            shown += 1


if __name__ == "__main__":
    main()
