"""Diagnose feature-space collapse under the TRAINING distribution.

Extracts features from real composites (caption+photo) — NOT the probe
distribution (single word + white). Reports:
  1. photo-feature pairwise cos (is photo space collapsed too?)
  2. word-feature pairwise cos from real captions (same word across imgs vs diff words)
  3. word-vs-photo cos alignment (is the CLIP alignment real or collapse-driven?)
"""
import argparse
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train_crossmodal_jepa import (CrossModalJEPA, build_composite,
                                   boxes_to_cell_masks, region_mask)

STOP = set("the a an of in on at to for and or but is are was were be been being "
           "with without from by as this that these those it its their his her my "
           "your our we they he she i you very into over under above below up down "
           "out off again further then once here there all any both each few more "
           "most other some such no nor not only own same so than too can will just "
           "don should now".split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tar_dir", required=True)
    p.add_argument("--n", type=int, default=300)
    args = p.parse_args()

    device = "cuda"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    margs = ck["args"]
    img_size = margs["img_size"]
    grid = img_size // margs["patch_size"]
    patch = margs["patch_size"]
    print(f"[diag] ckpt={args.ckpt}  grid={grid} hidden={margs['hidden']}", flush=True)

    model = CrossModalJEPA("convnext", img_size, margs["hidden"], margs["layers"],
                           margs["heads"], margs["patch_size"], margs["pred_depth"],
                           margs["ema_tau"]).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    tc = region_mask(grid, list(range(grid // 2))).to(device)
    pc = region_mask(grid, list(range(grid // 2, grid))).to(device)

    tars = sorted(Path(args.tar_dir).glob("*.tar"))[:5]
    samples = []
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
                    samples.append((cap, img_bytes))
                    if len(samples) >= args.n:
                        break
                except Exception:
                    continue
        tf.close()
        if len(samples) >= args.n:
            break
    print(f"[diag] loaded {len(samples)} samples", flush=True)

    photo_fs, text_fs = [], []
    word_feats = {}  # word -> list of feats

    with torch.no_grad():
        for cap, img_bytes in samples:
            img = Image.open(io.BytesIO(img_bytes))
            composite, boxes = build_composite(cap, img)
            t = torch.from_numpy(composite).float().permute(2, 0, 1) / 255.0
            t = ((t - 0.5) / 0.5).unsqueeze(0).to(device)
            f = model.encoder(t, return_map=True).float()  # (1, N, D)

            photo_fs.append(F.normalize(f[0, pc].mean(0), dim=0))
            text_fs.append(F.normalize(f[0, tc].mean(0), dim=0))

            words = cap.lower().split()
            words = [w.strip(".,!?;:'\"()[]") for w in words]
            if boxes and len(boxes) == len(words):
                masks = boxes_to_cell_masks(boxes, grid, patch).float().to(device)
                wf = torch.einsum("wn,nd->wd", masks, f[0]) / masks.sum(1, keepdim=True)
                wf = F.normalize(wf, dim=-1)
                for w, feat in zip(words, wf):
                    if w and w not in STOP and len(w) > 2:
                        word_feats.setdefault(w, []).append(feat)

    photo_fs = torch.stack(photo_fs)
    text_fs = torch.stack(text_fs)

    def pairwise_offdiag(M):
        sim = M @ M.T
        idx = torch.triu_indices(sim.size(0), sim.size(0), offset=1)
        return sim[idx[0], idx[1]]

    # 1. photo space
    pc_sim = pairwise_offdiag(photo_fs)
    print(f"\n[1] PHOTO pairwise cos:  mean={pc_sim.mean():.4f}  std={pc_sim.std():.4f}  "
          f"min={pc_sim.min():.4f}  max={pc_sim.max():.4f}", flush=True)

    # 2. text-global space
    tc_sim = pairwise_offdiag(text_fs)
    print(f"[2] TEXT-GLOBAL pairwise cos:  mean={tc_sim.mean():.4f}  std={tc_sim.std():.4f}", flush=True)

    # 3. word feats: intra-word (same word across imgs) vs inter-word (diff words)
    common = {w: torch.stack(v) for w, v in word_feats.items() if len(v) >= 5}
    print(f"\n[3] words with >=5 instances: {len(common)}", flush=True)
    if common:
        intra = []
        for w, fs in common.items():
            if fs.size(0) >= 2:
                intra.append(pairwise_offdiag(fs).mean().item())
        protos = {w: F.normalize(fs.mean(0), dim=0) for w, fs in common.items()}
        wnames = list(protos.keys())
        Wmat = torch.stack([protos[w] for w in wnames])
        inter = pairwise_offdiag(Wmat)
        print(f"    intra-word cos (same word, diff imgs): mean={np.mean(intra):.4f}  std={np.std(intra):.4f}", flush=True)
        print(f"    inter-word cos (word prototypes):      mean={inter.mean():.4f}  std={inter.std():.4f}", flush=True)
        print(f"    margin (intra - inter): {np.mean(intra) - inter.mean().item():+.4f}", flush=True)
        # show top distinctive words (lowest inter-cos to others)
        order = inter.view(len(wnames), -1).mean(0).argsort()
        print("    most distinctive words:", [(wnames[i], f'{inter.view(len(wnames),-1).mean(1)[i]:.3f}') for i in order[:5]])

    # 4. word-vs-photo alignment (from same composite — has leakage but informative)
    if common:
        # for each word proto, average cos with all photo feats
        align = []
        for w, proto in protos.items():
            cos = (photo_fs @ proto).mean().item()
            align.append(cos)
        print(f"\n[4] word-proto vs photo cos: mean={np.mean(align):.4f}  std={np.std(align):.4f}", flush=True)

    # 5. cross-check: word proto vs its own images' photo vs other images' photo
    if common:
        same_img_cos = []   # word feat vs photo of the SAME image
        diff_img_cos = []   # word feat vs photo of DIFFERENT images
        keys = list(common.keys())
        for w in keys[:10]:
            for i, feat in enumerate(word_feats[w][:5]):
                if i < len(photo_fs):
                    same_img_cos.append(F.cosine_similarity(feat, photo_fs[i], dim=0).item())
                    j = (i + 137) % len(photo_fs)
                    diff_img_cos.append(F.cosine_similarity(feat, photo_fs[j], dim=0).item())
        print(f"\n[5] word-vs-photo SAME image: {np.mean(same_img_cos):.4f}", flush=True)
        print(f"    word-vs-photo DIFF image: {np.mean(diff_img_cos):.4f}", flush=True)
        print(f"    gap (same - diff):        {np.mean(same_img_cos) - np.mean(diff_img_cos):+.4f}", flush=True)


if __name__ == "__main__":
    main()
