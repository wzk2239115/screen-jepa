import re

import numpy as np
import torch
from torch.utils.data import Dataset

from render import TextRenderer

DEFAULT_PARQUET = "/home/wzk/projects/screen-jepa/data/common_corpus_sample/common_corpus_1/subset_100_1.parquet"


def load_sentences(corpus_txt=None, parquet_path=DEFAULT_PARQUET, max_sentences=20000,
                   min_words=5, max_words=25, ascii_only=True):
    """Load and clean a list of sentences from a txt file, falling back to a
    local parquet text column. Filters by length and (optionally) ascii."""
    if corpus_txt:
        with open(corpus_txt, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    else:
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        text = "\n".join(df["text"].dropna().astype(str).tolist())

    cands = []
    for chunk in re.split(r"(?<=[.!?])\s+", text):
        c = " ".join(chunk.split())
        n = len(c.split())
        if not (min_words <= n <= max_words):
            continue
        if not (10 <= len(c) <= 160):
            continue
        if ascii_only and not c.isascii():
            continue
        cands.append(c)

    seen, out = set(), []
    for s in cands:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= max_sentences:
            break
    return out


def to_tensor(a):
    """(H,W,3) uint8 -> (3,H,W) float in [-1,1]."""
    t = torch.from_numpy(a).float().permute(2, 0, 1) / 255.0
    return (t - 0.5) / 0.5


class TextImageDataset(Dataset):
    """Yields (full, masked) image-view pairs of the same rendered sentence."""

    def __init__(self, sentences, img_size=224, font_size=16, mask_ratio=0.15, mask_min=1):
        self.sents = sentences
        self.r = TextRenderer(img_size=img_size, font_size=font_size)
        self.mask_ratio = mask_ratio
        self.mask_min = mask_min

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, i):
        full, boxes = self.r.render(self.sents[i])
        n = len(boxes)
        if n == 0:
            full = np.full((self.r.img_size, self.r.img_size, 3), 255, dtype=np.uint8)
            return to_tensor(full), to_tensor(full.copy())
        k = max(self.mask_min, int(round(n * self.mask_ratio)))
        k = min(k, n)
        idx = np.random.choice(n, size=k, replace=False)
        masked = self.r.mask_words(full, boxes, idx)
        return to_tensor(full), to_tensor(masked)


if __name__ == "__main__":
    sents = load_sentences(ascii_only=True, max_sentences=1000)
    print("sentences:", len(sents))
    for s in sents[:3]:
        print("  -", s)
    ds = TextImageDataset(sents)
    f, m = ds[0]
    print("full", tuple(f.shape), f.min().item(), f.max().item())
    print("masked", tuple(m.shape), "diff px:", int((f != m).any(0).sum().item()))
