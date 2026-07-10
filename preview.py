"""Render a preview montage of (full, masked) image pairs exactly as the network
sees them, using the same dataset pipeline. Run:

    python preview.py [--parquet_path ...] [--font_size 72] [--n 8] [--out preview.png]
"""
import argparse

import numpy as np
from PIL import Image

from dataset import TextImageDataset, load_sentences


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_txt", default=None)
    p.add_argument("--parquet_path",
                   default="/home/wzk/projects/screen-jepa/data/common_corpus_sample/common_corpus_1/subset_100_1.parquet")
    p.add_argument("--hf_dataset", default=None)
    p.add_argument("--hf_config", default=None)
    p.add_argument("--hf_split", default="train")
    p.add_argument("--language", default=None)
    p.add_argument("--ascii_only", type=int, default=1)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--font_size", type=int, default=72)
    p.add_argument("--mask_ratio", type=float, default=0.2)
    p.add_argument("--bg_augment", type=int, default=0)
    p.add_argument("--bg_block", type=int, default=0)
    p.add_argument("--font_augment", type=int, default=0)
    p.add_argument("--geom_strength", type=int, default=0)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--out", default="preview.png")
    args = p.parse_args()

    sents = load_sentences(args.corpus_txt, args.parquet_path, max_sentences=args.n * 20,
                           ascii_only=bool(args.ascii_only), hf_dataset=args.hf_dataset,
                           hf_config=args.hf_config, hf_split=args.hf_split, language=args.language)
    sents = sents[: args.n] if len(sents) >= args.n else sents
    ds = TextImageDataset(sents, img_size=args.img_size,
                          font_size=args.font_size, mask_ratio=args.mask_ratio,
                          bg_augment=bool(args.bg_augment),
                          bg_block=bool(args.bg_block),
                          font_augment=bool(args.font_augment),
                          geom_strength=args.geom_strength)

    S = args.img_size
    pad = 4
    rows = []
    for i in range(len(sents)):
        full_t, masked_t = ds[i]
        full = (((full_t.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8))
        masked = (((masked_t.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8))
        pair = np.full((S, 2 * S + pad, 3), 200, dtype=np.uint8)
        pair[:, :S] = full
        pair[:, S + pad:] = masked
        rows.append(pair)
    grid = np.concatenate(rows, axis=0) if rows else np.zeros((S, 2 * S, 3), np.uint8)
    Image.fromarray(grid).save(args.out)
    print(f"saved {args.out}: {len(rows)} rows x (full | masked), each {S}x{S}")
    for i, s in enumerate(sents):
        print(f"  [{i}] {s[:90]}")


if __name__ == "__main__":
    main()
