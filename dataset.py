import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from render import DEFAULT_FONT, TextRenderer, geom_augment, patchwork_bg

DEFAULT_PARQUET = "/home/wzk/projects/screen-jepa/data/common_corpus_sample/common_corpus_1/subset_100_1.parquet"

DEFAULT_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]


def _filter_sentence(c, min_words, max_words, ascii_only):
    c = " ".join(c.split())
    n = len(c.split())
    if not (min_words <= n <= max_words):
        return None
    if not (10 <= len(c) <= 160):
        return None
    if ascii_only and not c.isascii():
        return None
    return c


def _dedup(cands, max_sentences):
    seen, out = set(), []
    for s in cands:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= max_sentences:
            break
    return out


def load_sentences(corpus_txt=None, parquet_path=DEFAULT_PARQUET, max_sentences=20000,
                   min_words=5, max_words=25, ascii_only=True,
                   hf_dataset=None, hf_config=None, hf_split="train", language=None,
                   used=None):
    """Load and clean a list of sentences.

    Priority: corpus_txt > hf_dataset (streaming, uses HF cache) > local parquet.
    Streaming from a HF dataset stops as soon as max_sentences are collected.
    Optional `language` filters the dataset's 'language' column (e.g. 'English')."""
    cands = []

    if hf_dataset:
        from datasets import load_dataset

        ds = load_dataset(hf_dataset, hf_config, split=hf_split, streaming=True)
        for row in ds:
            if language is not None and row.get("language") != language:
                continue
            text = row.get("text")
            if not text:
                continue
            for chunk in re.split(r"(?<=[.!?])\s+", text):
                c = _filter_sentence(chunk, min_words, max_words, ascii_only)
                if c is not None:
                    cands.append(c)
            if len(cands) >= max_sentences * 4:
                break
        return _dedup(cands, max_sentences)

    if corpus_txt:
        with open(corpus_txt, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        for chunk in re.split(r"(?<=[.!?])\s+", text):
            c = _filter_sentence(chunk, min_words, max_words, ascii_only)
            if c is not None:
                cands.append(c)
        return _dedup(cands, max_sentences)

    import glob

    import pandas as pd

    if any(ch in parquet_path for ch in "*?["):
        paths = sorted(glob.glob(parquet_path, recursive=True))
    else:
        paths = [parquet_path]
    if not paths:
        raise FileNotFoundError(f"no parquet matched: {parquet_path}")

    target = max_sentences * 4
    cols = ["text"] + (["language"] if language is not None else [])
    for p in paths:
        if used is not None:
            used.append(p)
        df = pd.read_parquet(p, columns=cols)
        if language is not None and "language" in df.columns:
            df = df[df["language"] == language]
        for text in df["text"].dropna().astype(str):
            for chunk in re.split(r"(?<=[.!?])\s+", text):
                c = _filter_sentence(chunk, min_words, max_words, ascii_only)
                if c is not None:
                    cands.append(c)
        if len(cands) >= target:
            break
    return _dedup(cands, max_sentences)


def to_tensor(a):
    """(H,W,3) uint8 -> (3,H,W) float in [-1,1]."""
    t = torch.from_numpy(a).float().permute(2, 0, 1) / 255.0
    return (t - 0.5) / 0.5


GEOM_PRESETS = {
    1: dict(scale_lo=0.92, scale_hi=1.0, max_shear=1.0),   # light (zoom-out only, no rotation)
    2: dict(scale_lo=0.82, scale_hi=1.0, max_shear=2.5),   # medium
}


class TextImageDataset(Dataset):
    """Yields (full, masked[, cell_mask]) for a rendered sentence.

    cell_mask (B optional): (grid*grid) bool marking feature-grid cells that fall
    inside masked words' bboxes; used by the predictive JEPA objective."""

    def __init__(self, sentences, img_size=224, font_size=16, mask_ratio=0.15,
                 mask_min=1, grid=14, return_cell_mask=False, bg_augment=False,
                 font_augment=False, font_pool=None, geom_strength=0, bg_block=False):
        self.sents = sentences
        self.mask_ratio = mask_ratio
        self.mask_min = mask_min
        self.grid = grid
        self.return_cell_mask = return_cell_mask
        self.bg_augment = bg_augment
        self.bg_block = bg_block
        self.font_augment = font_augment
        self.geom_strength = geom_strength
        self.stride = img_size / grid
        pool = (font_pool.split(",") if font_pool else DEFAULT_FONTS)
        pool = [p for p in pool if os.path.exists(p)] or [DEFAULT_FONT]
        self.font_pool = pool if font_augment else [DEFAULT_FONT]
        self.renderers = [TextRenderer(img_size=img_size, font_size=font_size, font_path=p)
                          for p in self.font_pool]
        self.r = self.renderers[0]

    @staticmethod
    def _rand_bg():
        return tuple(int(np.random.randint(150, 256)) for _ in range(3))

    def _pick(self):
        return self.renderers[np.random.randint(len(self.renderers))]

    def __len__(self):
        return len(self.sents)

    def _cell_mask(self, boxes, idx, S):
        g = self.grid
        cm = torch.zeros(g, g, dtype=torch.bool)
        for i in idx:
            x0, y0, x1, y1 = boxes[i]
            c0 = int(min(max(x0 / self.stride, 0), g - 1))
            c1 = int(min(max((x1 - 1) / self.stride, 0), g - 1))
            r0 = int(min(max(y0 / self.stride, 0), g - 1))
            r1 = int(min(max((y1 - 1) / self.stride, 0), g - 1))
            cm[r0:r1 + 1, c0:c1 + 1] = True
        return cm.flatten()

    def __getitem__(self, i):
        sent = self.sents[i]
        r1 = self._pick()
        full, boxes = r1.render(sent)
        S = r1.img_size
        n = len(boxes)
        if n == 0:
            full = np.full((S, S, 3), 255, dtype=np.uint8)
            t = to_tensor(full)
            if self.return_cell_mask:
                return t, t.clone(), torch.zeros(self.grid * self.grid, dtype=torch.bool)
            return t, t.clone()
        k = max(self.mask_min, int(round(n * self.mask_ratio)))
        k = min(k, n)
        idx = np.random.choice(n, size=k, replace=False)
        if self.bg_augment or self.bg_block or self.font_augment or self.geom_strength:
            r2 = self._pick()
            if self.bg_block:
                bg_img = patchwork_bg(S)
                bg = (230, 230, 230)
            else:
                bg_img = None
                bg = self._rand_bg() if self.bg_augment else (255, 255, 255)
            alt, boxes2 = r2.render(sent, bg_color=bg, bg_img=bg_img)
            masked = r2.mask_words(alt, boxes2, idx, bg_color=bg, bg_img=bg_img)
            if self.geom_strength:
                masked = geom_augment(masked, bg_color=bg, **GEOM_PRESETS[self.geom_strength])
            cell_boxes = boxes2
        else:
            masked = r1.mask_words(full, boxes, idx)
            cell_boxes = boxes
        full_t, masked_t = to_tensor(full), to_tensor(masked)
        if self.return_cell_mask:
            return full_t, masked_t, self._cell_mask(cell_boxes, idx, S)
        return full_t, masked_t


if __name__ == "__main__":
    sents = load_sentences(ascii_only=True, max_sentences=1000)
    print("sentences:", len(sents))
    for s in sents[:3]:
        print("  -", s)
    ds = TextImageDataset(sents)
    f, m = ds[0]
    print("full", tuple(f.shape), f.min().item(), f.max().item())
    print("masked", tuple(m.shape), "diff px:", int((f != m).any(0).sum().item()))
