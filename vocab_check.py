"""Check whether the probe words (synonyms/antonyms/retrieval vocab) actually
appear in the training data. If they are OOD, the semantic/retrieval probes are
invalid (they test unseen words, not 'failed to learn').

Reads the same pinned parquet + language filter used for training, so it matches
exactly. Optionally point --datasource at outputs/<run>/datasource.txt.

Usage:
    python vocab_check.py \
        --parquet_path '/root/.cache/.../subset_100_[1-9].parquet' \
        --language English
"""
import argparse
import collections

from dataset import load_sentences
from probe import ANTONYMS, SYNONYMS
from retrieval import WORDS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_txt", default=None)
    p.add_argument("--parquet_path",
                   default="/home/wzk/projects/screen-jepa/data/common_corpus_sample/common_corpus_1/subset_100_1.parquet")
    p.add_argument("--hf_dataset", default=None)
    p.add_argument("--language", default="English")
    p.add_argument("--ascii_only", type=int, default=1)
    p.add_argument("--max_sentences", type=int, default=200000)
    p.add_argument("--datasource", default=None,
                   help="outputs/<run>/datasource.txt to reuse the exact pinned files")
    args = p.parse_args()

    if args.datasource:
        with open(args.datasource) as f:
            files = [l.strip() for l in f if l.strip()]
        pq = " ".join(files) if files else args.parquet_path
        # rebuild a glob only if single pattern; else load each
        sents = []
        import pandas as pd, re
        for fp in files:
            df = pd.read_parquet(fp, columns=["text", "language"]) if True else None
            if args.language:
                df = df[df["language"] == args.language]
            for t in df["text"].dropna().astype(str):
                for chunk in re.split(r"(?<=[.!?])\s+", t):
                    c = " ".join(chunk.split())
                    if 5 <= len(c.split()) <= 25:
                        sents.append(c.lower())
            if len(sents) > args.max_sentences:
                break
    else:
        sents = [s.lower() for s in load_sentences(
            parquet_path=args.parquet_path, max_sentences=args.max_sentences,
            ascii_only=bool(args.ascii_only), language=args.language)]

    # word frequency over training sentences
    freq = collections.Counter()
    for s in sents:
        for w in s.split():
            freq[w] += 1
    print(f"training sentences: {len(sents)}, unique tokens: {len(freq)}")

    def coverage(name, words):
        words = [w.lower() for w in words]
        present = [(w, freq.get(w, 0)) for w in words]
        seen = [w for w, c in present if c > 0]
        missing = [w for w, c in sorted(present, key=lambda x: x[1]) if c == 0]
        print(f"\n[{name}] {len(seen)}/{len(words)} present in training data")
        if missing:
            print(f"  MISSING ({len(missing)}): {', '.join(missing[:40])}")
        low = sorted([(w, c) for w, c in present if 0 < c <= 5], key=lambda x: x[1])
        if low:
            print(f"  RARE (<=5 occ): {', '.join(f'{w}({c})' for w,c in low[:20])}")
        return len(seen) / max(1, len(words))

    vocab_syn = [w for pr in SYNONYMS for w in pr]
    vocab_ant = [w for pr in ANTONYMS for w in pr]
    coverage("synonyms", vocab_syn)
    coverage("antonyms", vocab_ant)
    coverage("retrieval_words", WORDS)


if __name__ == "__main__":
    main()
