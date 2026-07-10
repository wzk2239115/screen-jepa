"""Fill-in (cloze) probe: does the predictor use context to prefer semantically
fitting words?

For a real training sentence containing a target word T:
  - mask T (erase its pixels)
  - context prediction: predictor(encoder(masked)) at T's cell -> a predicted
    feature vector (what the model thinks belongs there, from context alone)
  - for each candidate word (correct=T, related=syn/ant of T, random): composite
    the candidate into T's slot -> target_encoder(filled) at T's cell -> feature
  - score(candidate) = cosine(prediction, candidate_feature)

A model that learned context->meaning should rank: correct >= related > random.
For synonyms related>random = semantic signal; for antonyms related<random.
"""
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image, ImageDraw

from dataset import load_sentences, to_tensor
from probe import ANTONYMS, SYNONYMS
from render import TextRenderer


def find_sentences_with(word, sentences, k=3):
    out = []
    w = f" {word.lower()} "
    for s in sentences:
        if w in f" {s.lower()} ":
            out.append(s)
            if len(out) >= k:
                break
    return out


def cell_of(box, grid, img_size):
    x0, y0, x1, y1 = box
    stride = img_size / grid
    cx = int(min(max(((x0 + x1) / 2) / stride, 0), grid - 1))
    cy = int(min(max(((y0 + y1) / 2) / stride, 0), grid - 1))
    return cy * grid + cx


def composite(masked_img, word, box, font):
    pil = Image.fromarray(masked_img.copy())
    d = ImageDraw.Draw(pil)
    x0, y0, x1, y1 = box
    d.rectangle([x0 - 1, y0 - 1, x1 + 1, y1 + 1], fill=(255, 255, 255))
    d.text((x0, y0), word, fill=(0, 0, 0), font=font)
    return np.array(pil)


@torch.no_grad()
def predict_at(model, img_t, cell, device):
    """predictor output at `cell` from a (already masked) image."""
    ctx = model.encoder(img_t, return_map=True)  # (1,196,d)
    B, N, D = ctx.shape
    inp = ctx + model.pos
    m = torch.zeros(B, N, dtype=torch.bool, device=device)
    m[0, cell] = True
    inp = torch.where(m.unsqueeze(-1), model.mask_token.expand(B, N, D), inp)
    out = model.pred_norm(model.predictor(inp))
    return out[0, cell]


@torch.no_grad()
def target_at(model, img_t, cell, device):
    t = model.target_encoder(img_t, return_map=True)  # (1,196,d)
    return t[0, cell]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--parquet_path", required=True)
    p.add_argument("--language", default="English")
    p.add_argument("--ascii_only", type=int, default=1)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--font_path", default="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    p.add_argument("--max_sentences", type=int, default=200000)
    p.add_argument("--per_word", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ca = ckpt["args"]
    assert ca.get("objective") == "predictive", "fill-in probe needs a predictive checkpoint"
    from pred_model import PredictiveJEPA
    model = PredictiveJEPA(arch=ca.get("arch", "convnext"), img_size=args.img_size,
                           hidden=ca["hidden"], layers=ca["layers"], heads=ca["heads"],
                           patch=ca.get("patch_size", 16),
                           pred_depth=ca.get("pred_depth", 4)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    grid = model.grid

    renderer = TextRenderer(img_size=args.img_size, font_size=ca.get("font_size", 72),
                            font_path=args.font_path)
    sents = [s for s in load_sentences(parquet_path=args.parquet_path,
                                       max_sentences=args.max_sentences,
                                       ascii_only=bool(args.ascii_only),
                                       language=args.language)]
    print(f"[data] {len(sents)} sentences; grid={grid}")

    def run_pairs(pairs, kind):
        correct_s, related_s, random_s = [], [], []
        pool = [w for pr in pairs for w in pr]
        used = 0
        for a, b in pairs:
            sents_a = find_sentences_with(a, sents, args.per_word)
            for s in sents_a:
                wlist = s.split()
                idxs = [i for i, w in enumerate(wlist) if w.lower() == a.lower()]
                if not idxs:
                    continue
                ti = idxs[0]
                img, boxes = renderer.render(s)
                if ti >= len(boxes):
                    continue
                box = boxes[ti]
                cell = cell_of(box, grid, args.img_size)
                masked = renderer.mask_words(img, boxes, [ti])
                pred = predict_at(model, to_tensor(masked).unsqueeze(0).to(device), cell, device)
                rw = random.choice([w for w in pool if w not in (a, b)])
                cands = {"correct": a, kind: b, "random": rw}
                feats = {}
                for role, w in cands.items():
                    filled = composite(masked, w, box, renderer.last_font)
                    feats[role] = target_at(model, to_tensor(filled).unsqueeze(0).to(device), cell, device)
                for role in cands:
                    sc = Fn.cosine_similarity(pred, feats[role], dim=-1).item()
                    {"correct": correct_s, kind: related_s, "random": random_s}[role].append(sc)
                used += 1
        if not correct_s:
            print(f"[{kind}] no usable sentences"); return
        print(f"[{kind}] n={len(correct_s)}  "
              f"correct={np.mean(correct_s):.3f}  {kind}={np.mean(related_s):.3f}  "
              f"random={np.mean(random_s):.3f}  "
              f"({kind}-random={np.mean(related_s)-np.mean(random_s):+.3f}, "
              f"correct-random={np.mean(correct_s)-np.mean(random_s):+.3f})")

    run_pairs(SYNONYMS, "synonym")
    run_pairs(ANTONYMS, "antonym")


if __name__ == "__main__":
    main()
