"""Zero-shot image→word retrieval probe for TC-JEPA.

Encodes test images with the (EMA) target encoder, builds word prototypes
from training images, and checks whether each image's caption words are
retrievable via cosine similarity.

Usage:
    python probe_tc_jepa.py --ckpt outputs/tcjepa_v1/epoch99.pt \
        --tar_dir /path/to/recap-datacomp-384-1M --num_tars 5
"""

import io
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

STOP = set("a an the of in on at to for with and or but is are was were be been "
           "being this that these those it its as by from up down out over under "
           "into onto off above below near far very more most much many some any "
           "all no not can could should would will do does did has have had "
           "which what who whom whose where when why how than then there here "
           "about through during before after between among across along".split())


def clean_word(w):
    w = re.sub(r"[^a-z]", "", w.lower())
    return w if len(w) > 2 and w not in STOP else None


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ck["args"]

    if args.get("arch") == "clip":
        from train_clip import CLIPModel
        model = CLIPModel(
            hidden=args["hidden"], img_size=args["img_size"],
            patch=16, text_layers=args["text_layers"], text_heads=args["text_heads"],
        )
        model.load_state_dict(ck["model"])
        model = model.to(device).eval()
        model._is_clip = True
    else:
        from tc_jepa import TCJEPA
        model = TCJEPA(
            img_size=args["img_size"], patch_size=args["patch_size"],
            encoder_dim=args["encoder_dim"], encoder_depth=args["encoder_depth"],
            encoder_heads=args["encoder_heads"],
            pred_dim=args["pred_dim"], pred_depth=args["pred_depth"],
            pred_heads=args["pred_heads"],
            t5_model=args["t5_model"],
        )
        model.load_state_dict(ck["model"], strict=False)
        model = model.to(device).eval()
        model._is_clip = False

    print(f"[model] loaded {ckpt_path} ({'CLIP' if model._is_clip else 'TC-JEPA'})", flush=True)
    return model, args


def encode_images(model, images, device, batch_size=64):
    """Encode images → global features (N, D) via avg pooling."""
    feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.stack(images[i:i+batch_size]).to(device)
            if getattr(model, "_is_clip", False):
                z = model.encode_image(batch)       # (B, D) already global + normalized
            else:
                z = model.target_encoder(batch)     # (B, 196, D)
                z = z.mean(dim=1)                    # (B, D)
            feats.append(F.normalize(z.cpu(), dim=-1))
    return torch.cat(feats, dim=0)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tar_dir", required=True)
    p.add_argument("--num_tars", type=int, default=5)
    p.add_argument("--n_proto", type=int, default=2000, help="images for building word prototypes")
    p.add_argument("--n_test", type=int, default=500, help="held-out images for evaluation")
    p.add_argument("--min_count", type=int, default=5, help="min occurrences for a word to have a prototype")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, margs = load_model(args.ckpt, device)

    # ---- load data ----
    tars = sorted(str(p) for p in Path(args.tar_dir).glob("*.tar"))[:args.num_tars]
    transform = transforms.Compose([
        transforms.Resize((margs["img_size"], margs["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    samples = []
    for tp in tars:
        tf = tarfile.open(tp)
        for m in tf.getmembers():
            if m.name.endswith(".jpg"):
                samples.append((tp, m.name))
        tf.close()
    print(f"[data] {len(samples)} images", flush=True)

    import random
    random.seed(42)
    random.shuffle(samples)
    proto_samples = samples[:args.n_proto]
    test_samples = samples[args.n_proto : args.n_proto + args.n_test]

    # ---- helper to load image + caption ----
    _tar_cache = {}
    def get_tar(tp):
        if tp not in _tar_cache:
            _tar_cache[tp] = tarfile.open(tp)
        return _tar_cache[tp]

    def load_pair(tp, name):
        tf = get_tar(tp)
        img = Image.open(io.BytesIO(tf.extractfile(name).read())).convert("RGB")
        cap = json.loads(tf.extractfile(name.replace(".jpg", ".json")).read())["caption"]
        return transform(img), cap

    # ---- build word prototypes ----
    print(f"[proto] encoding {len(proto_samples)} images...", flush=True)
    proto_imgs, proto_caps = [], []
    for tp, name in proto_samples:
        try:
            img, cap = load_pair(tp, name)
            proto_imgs.append(img)
            proto_caps.append(cap)
        except Exception:
            continue

    proto_feats = encode_images(model, proto_imgs, device)  # (Np, D)

    # build word → prototype
    word_feats = defaultdict(list)
    for feat, cap in zip(proto_feats, proto_caps):
        words = set()
        for w in cap.split():
            cw = clean_word(w)
            if cw:
                words.add(cw)
        for w in words:
            word_feats[w].append(feat)

    word_list = sorted(w for w, fs in word_feats.items() if len(fs) >= args.min_count)
    if not word_list:
        print("[ERROR] no words with enough occurrences", flush=True)
        return
    Wmat = torch.stack([
        F.normalize(torch.stack(word_feats[w]).mean(0), dim=-1)
        for w in word_list
    ])  # (num_words, D)
    print(f"[proto] {len(word_list)} word prototypes from {len(proto_imgs)} images", flush=True)

    # ---- evaluate ----
    print(f"[eval] encoding {len(test_samples)} test images...", flush=True)
    test_imgs, test_caps = [], []
    for tp, name in test_samples:
        try:
            img, cap = load_pair(tp, name)
            test_imgs.append(img)
            test_caps.append(cap)
        except Exception:
            continue

    test_feats = encode_images(model, test_imgs, device)  # (Nt, D)

    # retrieval
    sims = test_feats @ Wmat.T  # (Nt, num_words)
    top1, top5, top10, mrr = 0, 0, 0, 0.0
    for i in range(len(test_feats)):
        gt_words = set()
        for w in test_caps[i].split():
            cw = clean_word(w)
            if cw and cw in set(word_list):
                gt_words.add(cw)
        if not gt_words:
            continue

        order = sims[i].argsort(descending=True)
        ranked = [word_list[j] for j in order]

        hit5 = any(w in ranked[:5] for w in gt_words)
        hit10 = any(w in ranked[:10] for w in gt_words)
        hit1 = any(w in ranked[:1] for w in gt_words)

        top1 += hit1
        top5 += hit5
        top10 += hit10

        # MRR: reciprocal rank of first hit
        for rank, w in enumerate(ranked):
            if w in gt_words:
                mrr += 1.0 / (rank + 1)
                break

    n = len(test_feats)
    print(f"\n=== Image→Word Retrieval (n_test={n}, n_words={len(word_list)}) ===")
    print(f"top-1:  {top1}/{n} = {top1/n:.3f}   (random={1.0/len(word_list):.4f})")
    print(f"top-5:  {top5}/{n} = {top5/n:.3f}   (random={5.0/len(word_list):.4f})")
    print(f"top-10: {top10}/{n} = {top10/n:.3f}   (random={10.0/len(word_list):.4f})")
    print(f"MRR:    {mrr/n:.3f}")

    # ---- qualitative examples ----
    print(f"\n--- Qualitative (first 10) ---")
    for i in range(min(10, len(test_feats))):
        gt = [clean_word(w) for w in test_caps[i].split()]
        gt = [w for w in gt if w and w in set(word_list)]
        order = sims[i].argsort(descending=True)
        top = [word_list[j] for j in order[:5]]
        print(f"  GT: {gt[:5]}  |  top5: {top}")


if __name__ == "__main__":
    main()
