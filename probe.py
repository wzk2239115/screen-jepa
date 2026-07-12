"""Synonym / antonym similarity probe.

Tests whether the pixel-trained encoder encodes word meaning: render each word
as an image, encode it, and compare cosine similarity of synonym pairs vs
antonym pairs vs random unrelated pairs.

A model that "understands meaning from pixels" should show:
    cos(synonyms) and cos(antonyms)  >>  cos(random pairs)
(antonyms often score ~synonyms in distributional spaces since both are
semantically related / substitutable, so the key signal is both > random.)

Usage:
    python probe.py --ckpt outputs/run_en/epoch49.pt
"""
import argparse
import itertools
import random

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

from dataset import to_tensor
from model import TextJEPA
from render import TextRenderer

SYNONYMS = [
    ("happy", "glad"), ("big", "large"), ("small", "tiny"), ("fast", "quick"),
    ("smart", "clever"), ("begin", "start"), ("end", "finish"), ("buy", "purchase"),
    ("help", "assist"), ("rich", "wealthy"), ("poor", "needy"), ("strong", "powerful"),
    ("weak", "feeble"), ("beautiful", "pretty"), ("funny", "hilarious"), ("sad", "unhappy"),
    ("angry", "furious"), ("calm", "peaceful"), ("easy", "simple"), ("hard", "difficult"),
    ("important", "significant"), ("old", "ancient"), ("new", "fresh"), ("correct", "right"),
    ("wrong", "incorrect"), ("increase", "grow"), ("decrease", "reduce"), ("show", "display"),
    ("hide", "conceal"), ("stop", "halt"), ("idea", "concept"), ("job", "work"),
    ("friend", "companion"), ("enemy", "foe"), ("truth", "fact"), ("lie", "falsehood"),
    ("danger", "peril"), ("begin", "commence"), ("fast", "rapid"), ("slow", "sluggish"),
]

ANTONYMS = [
    ("hot", "cold"), ("up", "down"), ("good", "bad"), ("day", "night"),
    ("open", "close"), ("win", "lose"), ("rich", "poor"), ("strong", "weak"),
    ("big", "small"), ("fast", "slow"), ("happy", "sad"), ("love", "hate"),
    ("right", "wrong"), ("light", "dark"), ("old", "new"), ("yes", "no"),
    ("begin", "end"), ("alive", "dead"), ("wet", "dry"), ("full", "empty"),
    ("high", "low"), ("long", "short"), ("wide", "narrow"), ("thick", "thin"),
    ("safe", "dangerous"), ("easy", "hard"), ("increase", "decrease"), ("enter", "exit"),
    ("remember", "forget"), ("accept", "reject"), ("arrive", "depart"), ("create", "destroy"),
    ("attack", "defend"), ("push", "pull"), ("rise", "fall"), ("laugh", "cry"),
]


@torch.no_grad()
def encode_words(words, model, renderer, device, n_aug=4, mode="mean"):
    """Encode each unique word from n_aug dense renders; average embeddings.
    mode: 'mean' = pooled vector; 'flatten' = flattened spatial feature map
    (exposes spatial word signal hidden by mean-pool)."""
    uniq = sorted(set(words))
    emb = {}
    for w in uniq:
        xs = []
        for _ in range(n_aug):
            img, _ = renderer.render(w)
            xs.append(to_tensor(img))
        x = torch.stack(xs).to(device)
        with torch.no_grad():
            if mode == "flatten" and hasattr(model, "encoder") and \
                    hasattr(model.encoder, "feature_dim"):
                z = model.encoder(x, return_map=True).float().flatten(1)
            else:
                z = model.encode(x).float()
        emb[w] = z.mean(0)
        emb[w] = emb[w] / (emb[w].norm() + 1e-8)
    return emb


def pair_sims(pairs, emb):
    sims = []
    for a, b in pairs:
        if a in emb and b in emb:
            sims.append(float(emb[a] @ emb[b]))
    return sims


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--font_path", default="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    p.add_argument("--mode", default="mean", choices=["mean", "flatten"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed); random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ca = ckpt["args"]
    if ca.get("objective") == "predictive":
        from pred_model import PredictiveJEPA
        model = PredictiveJEPA(arch=ca.get("arch", "convnext"), img_size=args.img_size,
                               hidden=ca["hidden"], layers=ca["layers"], heads=ca["heads"],
                               patch=ca.get("patch_size", 16),
                               pred_depth=ca.get("pred_depth", 4)).to(device)
    elif ca.get("objective") == "wordpred":
        from train_wordpred import WordPred
        vsz = len(ckpt.get("vocab") or []) or ca.get("vocab_size", 10000)
        model = WordPred("convnext", args.img_size, ca["hidden"], ca["layers"], ca["heads"],
                         ca.get("patch_size", 16), vocab_size=vsz).to(device)
    else:
        model = TextJEPA(ca["hidden"], ca["layers"], ca["heads"], ca.get("mlp_dim", 3072),
                         img_size=args.img_size, embed_dim=ca.get("embed_dim", 0) or ca["hidden"],
                         patch=ca.get("patch_size", 16), arch=ca.get("arch", "vit")).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    renderer = TextRenderer(img_size=args.img_size, font_size=ca.get("font_size", 72),
                            font_path=args.font_path)

    all_words = [w for pr in SYNONYMS + ANTONYMS for w in pr]
    emb = encode_words(all_words, model, renderer, device, mode=args.mode)
    print(f"=== word-image semantic probe  [mode={args.mode}] ===")

    syn = pair_sims(SYNONYMS, emb)
    ant = pair_sims(ANTONYMS, emb)
    pool = list(emb.values())
    rnd = []
    for _ in range(max(len(syn), len(ant)) * 2):
        i, j = np.random.choice(len(pool), 2, replace=False)
        rnd.append(float(pool[i] @ pool[j]))

    def stat(s):
        return f"mean={np.mean(s):.3f} std={np.std(s):.3f} n={len(s)}"

    print(f"synonyms : {stat(syn)}")
    print(f"antonyms : {stat(ant)}")
    print(f"random   : {stat(rnd)}")
    print(f"syn - random = {np.mean(syn)-np.mean(rnd):+.3f}")
    print(f"ant - random = {np.mean(ant)-np.mean(rnd):+.3f}")
    print(f"syn - ant    = {np.mean(syn)-np.mean(ant):+.3f}")

    rows = [("syn", a, b, sim) for (a, b), sim in zip(SYNONYMS, syn)]
    rows += [("ant", a, b, sim) for (a, b), sim in zip(ANTONYMS, ant)]
    rows.sort(key=lambda r: r[3])
    print("\nlowest-similarity pairs:")
    for k, a, b, s in rows[:8]:
        print(f"  [{k}] {a:12s} {b:12s} {s:+.3f}")
    print("highest-similarity pairs:")
    for k, a, b, s in rows[-8:]:
        print(f"  [{k}] {a:12s} {b:12s} {s:+.3f}")

    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["type", "a", "b", "cos"])
            w.writerows(rows)
        print(f"\nsaved per-pair sims -> {args.out}")


if __name__ == "__main__":
    main()
