"""Print per-epoch training (and val) metrics from log.csv / val_log.csv.

Usage:
    python show_log.py                       # default outputs/run2
    python show_log.py --dir outputs/run_en
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="outputs/run2")
    args = p.parse_args()

    d = Path(args.dir)
    log = d / "log.csv"
    if not log.exists():
        raise SystemExit(f"not found: {log}")
    df = pd.read_csv(log)

    print(f"=== train  [{log}]  {len(df)} rows ===")
    print(f"{'ep':>3} {'steps':>6} {'loss':>8} {'inv':>8} {'reg':>8} {'std':>6} {'dead':>6} {'cos':>6}")
    for ep, sub in df.groupby("epoch"):
        print(f"{ep:>3} {len(sub):>6} {sub.loss.mean():>8.4f} {sub.inv.mean():>8.4f} "
              f"{sub.reg.mean():>8.4f} {sub.z_std.mean():>6.3f} {sub.dead_dim.mean():>6.3f} "
              f"{sub.cos.iloc[-1]:>6.3f}")

    val = d / "val_log.csv"
    if val.exists():
        v = pd.read_csv(val)
        print(f"\n=== val  [{val}] ===")
        print(f"{'ep':>3} {'inv':>8} {'std':>6} {'dead':>6} {'cos_same':>9} {'cos_diff':>9} {'margin':>8}")
        for _, r in v.iterrows():
            print(f"{int(r.epoch):>3} {r.inv:>8.4f} {r.z_std:>6.3f} {r.dead_dim:>6.3f} "
                  f"{r.cos_same:>9.3f} {r.cos_diff:>9.3f} {r.margin:>8.3f}")


if __name__ == "__main__":
    main()
