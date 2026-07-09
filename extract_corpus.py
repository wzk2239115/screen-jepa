"""One-time extraction of a FIXED English sentence set from a pinned set of
common_corpus parquet files, so every arch run uses identical data regardless
of the ongoing download.

Usage:
    python extract_corpus.py \
        --parquet_path '/root/.cache/.../snapshots/<HASH>/common_corpus_1/subset_100_[1-9].parquet' \
        --language English --max_sentences 500000 \
        --out data/en_sentences.txt

Then train all arches with:  --corpus_txt data/en_sentences.txt
"""
import argparse
from pathlib import Path

from dataset import load_sentences


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet_path", required=True,
                   help="PINNED glob, e.g. '.../subset_100_[1-9].parquet' (do NOT use ***)")
    p.add_argument("--language", default="English")
    p.add_argument("--ascii_only", type=int, default=1)
    p.add_argument("--max_sentences", type=int, default=500000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    sents = load_sentences(parquet_path=args.parquet_path,
                           max_sentences=args.max_sentences,
                           ascii_only=bool(args.ascii_only),
                           language=args.language)
    if len(sents) < 1000:
        raise RuntimeError(f"only {len(sents)} sentences extracted; check path/language")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in sents:
            f.write(s + "\n")
    print(f"wrote {len(sents)} sentences -> {out}")


if __name__ == "__main__":
    main()
