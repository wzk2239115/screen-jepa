"""Noun-level semantic probe (more meaningful than syn/ant adjectives).

Validates: did the model ground noun features to image content?
  1. Noun clustering: do same-category nouns (dog/cat/horse = animals)
     have higher cosine than cross-category (dog/car/house)?
  2. Zero-shot classification: given an image containing a noun, can the
     model pick the right noun from the list (CLIP-style)?
"""
import argparse
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from train_crossmodal_jepa import CrossModalJEPA, DEFAULT_FONT, W, H, region_mask

NOUN_CATS = {
    "animal":  ["dog", "cat", "horse", "bird", "cow", "sheep"],
    "vehicle": ["car", "truck", "bus", "boat", "plane", "bike"],
    "food":    ["pizza", "cake", "bread", "fruit", "salad", "soup"],
    "scene":   ["tree", "mountain", "river", "beach", "sunset", "forest"],
    "person":  ["man", "woman", "child", "baby", "people"],
    "building": ["house", "tower", "church", "castle", "bridge"],
}


@torch.no_grad()
def word_emb(model, word, device, img_size, grid):
    """Render a single word centered in text region (lower half white).
    Average over 3 font sizes. Return L2-normalized text-region feature."""
    half = img_size // 2
    xs = []
    for fs in (40, 56, 72):
        block = Image.new("RGB", (img_size, half), (255, 255, 255))
        d = ImageDraw.Draw(block)
        font = ImageFont.truetype(DEFAULT_FONT, fs)
        d.text((img_size / 2, half / 2), word, fill=(0, 0, 0), font=font, anchor="mm")
        comp = np.concatenate([np.array(block),
                               np.full((half, img_size, 3), 255, np.uint8)], axis=0)
        t = torch.from_numpy(comp).float().permute(2, 0, 1) / 255.0
        xs.append((t - 0.5) / 0.5)
    f = model.encoder(torch.stack(xs).to(device), return_map=True).float()
    tc = region_mask(grid, list(range(grid // 2))).to(device)
    return F.normalize(f[:, tc].mean(1).mean(0), dim=0)


@torch.no_grad()
def image_emb(model, img_pil, device, img_size, grid):
    """Photo region feature from a composite (upper half white, lower half image)."""
    half = img_size // 2
    photo = img_pil.convert("RGB").resize((W, H))
    comp = np.concatenate([np.full((half, W, 3), 255, np.uint8),
                           np.array(photo)], axis=0)
    t = torch.from_numpy(comp).float().permute(2, 0, 1) / 255.0
    t = (t - 0.5) / 0.5
    f = model.encoder(t.unsqueeze(0).to(device), return_map=True).float()
    pc = region_mask(grid, list(range(grid // 2, grid))).to(device)
    return F.normalize(f[0, pc].mean(0), dim=0)


def collect_images(tar_dir, nouns, per_word=15, max_tars=30):
    """Scan tars for images whose caption contains each noun (exact word match)."""
    tars = sorted(Path(tar_dir).glob("*.tar"))[:max_tars]
    found = {w: [] for w in nouns}
    for tp in tars:
        try:
            tf = tarfile.open(tp)
        except Exception:
            continue
        for m in tf.getmembers():
            if not m.name.endswith(".jpg"):
                continue
            try:
                jf = tf.extractfile(m.name.replace(".jpg", ".json"))
                cap = json.loads(jf.read())["caption"].lower()
                words = set(cap.split())
                img_bytes = tf.extractfile(m.name).read()
                for w in nouns:
                    if w in words and len(found[w]) < per_word:
                        found[w].append(img_bytes)
            except Exception:
                continue
        tf.close()
        if all(len(v) >= per_word for v in found.values()):
            break
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tar_dir", required=True)
    p.add_argument("--per_word", type=int, default=15)
    args = p.parse_args()

    device = "cuda"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    margs = ck["args"]
    img_size = margs["img_size"]
    grid = img_size // margs["patch_size"]
    print(f"[probe] ckpt={args.ckpt}  img={img_size} grid={grid} hidden={margs['hidden']}", flush=True)

    model = CrossModalJEPA("convnext", img_size, margs["hidden"], margs["layers"],
                           margs["heads"], margs["patch_size"], margs["pred_depth"],
                           margs["ema_tau"]).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    nouns = [w for ws in NOUN_CATS.values() for w in ws]

    # === 1. noun clustering: same-cat vs diff-cat cosine ===
    print("=== noun embeddings ===", flush=True)
    E = {w: word_emb(model, w, device, img_size, grid) for w in nouns}

    same, diff = [], []
    cats = list(NOUN_CATS.items())
    for ci, (cat1, ws1) in enumerate(cats):
        for i in range(len(ws1)):
            for j in range(i + 1, len(ws1)):
                same.append(float((E[ws1[i]] * E[ws1[j]]).sum()))
        for cat2, ws2 in cats[ci + 1:]:
            for w1 in ws1:
                for w2 in ws2:
                    diff.append(float((E[w1] * E[w2]).sum()))
    sm, dm = float(np.mean(same)), float(np.mean(diff))
    print(f"[cluster] same-cat={sm:.4f}  diff-cat={dm:.4f}  margin={sm - dm:+.4f}", flush=True)
    print("per-category same-cat cos:")
    for cat, ws in NOUN_CATS.items():
        vals = [float((E[ws[i]] * E[ws[j]]).sum())
                for i in range(len(ws)) for j in range(i + 1, len(ws))]
        if vals:
            print(f"  {cat:10s}: {np.mean(vals):.4f}")

    # === 2. zero-shot image->noun classification ===
    print(f"\n=== collecting images (per_word={args.per_word}) ===", flush=True)
    found = collect_images(args.tar_dir, nouns, args.per_word)
    for w in nouns:
        print(f"  {w:12s}: {len(found[w])} imgs")

    word_list = [w for w in nouns if len(found[w]) >= 3]
    if not word_list:
        print("[probe] not enough images collected; abort zero-shot")
        return
    Wmat = torch.stack([E[w] for w in word_list])

    print(f"\n=== zero-shot classification ({len(word_list)} nouns) ===", flush=True)
    correct, total = 0, 0
    per_wc = {w: [0, 0] for w in word_list}
    confusion = {}
    for wi, w in enumerate(word_list):
        for img_bytes in found[w]:
            img = Image.open(io.BytesIO(img_bytes))
            img_f = image_emb(model, img, device, img_size, grid)
            sims = Wmat @ img_f
            pred = word_list[int(sims.argmax())]
            per_wc[w][1] += 1
            if pred == w:
                correct += 1
                per_wc[w][0] += 1
            else:
                confusion[(w, pred)] = confusion.get((w, pred), 0) + 1
            total += 1
    acc = correct / max(total, 1)
    rand = 1.0 / len(word_list)
    print(f"zero-shot acc: {correct}/{total} = {acc:.3f}  (random={rand:.3f})", flush=True)
    print("per-word accuracy:")
    for w in word_list:
        c, t = per_wc[w]
        print(f"  {w:12s}: {c}/{t} = {c / max(t, 1):.2f}")
    if confusion:
        print("top confusions (truth->pred):")
        for (gt, pd), n in sorted(confusion.items(), key=lambda x: -x[1])[:8]:
            print(f"  {gt:12s} -> {pd:12s}: {n}")


if __name__ == "__main__":
    main()
