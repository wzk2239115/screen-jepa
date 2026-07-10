"""Retrieval probe: does the encoder encode WORD IDENTITY?

Render each word many times with light rendering jitter, encode, then check
whether a render's nearest neighbor is another render of the SAME word.
Reports top-1 retrieval accuracy and same/diff cosine gap.

This is a more robust measure of word-identity encoding than the syn/ant
cosine probe (it does not depend on semantic relatedness, only on whether the
embedding can tell word A from word B).

Usage:
    python retrieval.py --ckpt outputs/pred_convnext/epoch29.pt --words 200
"""
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image

from dataset import to_tensor
from model import TextJEPA
from render import TextRenderer

WORDS = [
    "time", "year", "people", "way", "day", "man", "thing", "woman", "life",
    "child", "world", "school", "state", "family", "student", "group", "country",
    "problem", "hand", "part", "place", "case", "week", "company", "system",
    "program", "question", "work", "government", "number", "night", "point",
    "home", "water", "room", "mother", "area", "money", "story", "fact",
    "month", "lot", "right", "study", "book", "eye", "job", "word", "business",
    "issue", "side", "kind", "head", "house", "service", "friend", "father",
    "power", "hour", "game", "line", "end", "member", "law", "car", "city",
    "community", "name", "president", "team", "minute", "idea", "kid", "body",
    "information", "back", "parent", "face", "others", "level", "office", "door",
    "health", "person", "art", "war", "history", "party", "result", "change",
    "morning", "reason", "research", "girl", "guy", "moment", "air", "teacher",
    "force", "education", "foot", "boy", "age", "policy", "process", "music",
    "market", "sense", "nation", "plan", "college", "interest", "death",
    "experience", "effect", "use", "class", "control", "care", "field",
    "development", "role", "effort", "rate", "heart", "drug", "leader", "light",
    "voice", "wife", "police", "mind", "price", "report", "decision", "son",
    "hope", "view", "relationship", "town", "road", "arm", "source", "sound",
    "page", "century", "evidence", "page", "truth", "fire", "future", "past",
    "happy", "sad", "angry", "calm", "brave", "scared", "tired", "eager",
    "beautiful", "ugly", "huge", "tiny", "ancient", "modern", "simple", "complex",
    "fast", "slow", "strong", "weak", "rich", "poor", "loud", "quiet",
]


def jitter_render(renderer, word, fs_caps):
    fs = random.choice(fs_caps)
    r = TextRenderer(img_size=renderer.img_size, font_size=fs, font_path=renderer.font_path)
    img, _ = r.render(word)
    dx = random.randint(-8, 8)
    dy = random.randint(-4, 4)
    pil = Image.fromarray(img)
    canvas = Image.new("RGB", pil.size, (255, 255, 255))
    canvas.paste(pil, (dx, dy + 8))
    return np.array(canvas)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--words", type=int, default=150)
    p.add_argument("--font_path", default="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ca = ckpt["args"]
    if ca.get("objective") == "predictive":
        from pred_model import PredictiveJEPA
        model = PredictiveJEPA(arch=ca.get("arch", "convnext"), img_size=args.img_size,
                               hidden=ca["hidden"], layers=ca["layers"], heads=ca["heads"],
                               patch=ca.get("patch_size", 16),
                               pred_depth=ca.get("pred_depth", 4)).to(device)
    else:
        model = TextJEPA(ca["hidden"], ca["layers"], ca["heads"], ca.get("mlp_dim", 3072),
                         img_size=args.img_size, embed_dim=ca.get("embed_dim", 0) or ca["hidden"],
                         patch=ca.get("patch_size", 16), arch=ca.get("arch", "vit")).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    renderer = TextRenderer(img_size=args.img_size, font_size=ca.get("font_size", 72),
                            font_path=args.font_path)
    fs_caps = [56, 64, 72, 80]

    words = WORDS[: args.words]
    embs, labels = [], []
    for wi, w in enumerate(words):
        for _ in range(args.repeats):
            img = jitter_render(renderer, w, fs_caps)
            embs.append(to_tensor(img))
            labels.append(wi)
    X = torch.stack(embs).to(device)
    with torch.autocast("cuda", enabled=device == "cuda", dtype=torch.bfloat16):
        Z = model.encode(X).float()
    Z = Fn.normalize(Z, dim=-1).cpu()
    labels = torch.tensor(labels)

    sims = Z @ Z.t()
    torch.diagonal(sims).fill_(-1)
    nn = sims.argmax(dim=1)
    top1 = (labels[nn] == labels).float().mean().item()

    same, diff = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            s = float(sims[i, j])
            (same if labels[i] == labels[j] else diff).append(s)
    same_m, diff_m = np.mean(same), np.mean(diff)

    print(f"=== retrieval probe  (words={len(words)} repeats={args.repeats} n={len(labels)}) ===")
    print(f"top-1 NN accuracy   : {top1:.3f}   (chance ~= {1/len(words):.4f})")
    print(f"cos same-word       : {same_m:.3f}")
    print(f"cos different-word  : {diff_m:.3f}")
    print(f"gap (same - diff)   : {same_m-diff_m:+.3f}")
    print(f"emb std             : {Z.std(0).mean().item():.3f}")


if __name__ == "__main__":
    main()
