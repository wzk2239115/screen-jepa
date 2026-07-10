"""Print per-epoch training (and val) metrics from log.csv / val_log.csv.

Usage:
    python show_log.py --dir outputs/pred_convnext
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

    stat = "pred" if "pred" in df.columns else "inv"
    stdcol = "std" if "std" in df.columns else "z_std"

    print(f"=== train  [{log}]  {len(df)} rows ===")
    print(f"{'ep':>3} {'steps':>6} {'loss':>8} {stat:>8} {'reg':>8} {'std':>6} {'dead':>6} {'cos':>6}")
    for ep, sub in df.groupby("epoch"):
        regv = sub.reg.mean() if "reg" in sub.columns else 0.0
        print(f"{ep:>3} {len(sub):>6} {sub.loss.mean():>8.4f} {sub[stat].mean():>8.4f} "
              f"{regv:>8.4f} {sub[stdcol].mean():>6.3f} {sub.dead_dim.mean():>6.3f} "
              f"{sub.cos.iloc[-1]:>6.3f}")

    val = d / "val_log.csv"
    if val.exists():
        v = pd.read_csv(val)
        vloss = "loss" if "loss" in v.columns else "inv"
        vstd = "std" if "std" in v.columns else "z_std"
        print(f"\n=== val  [{val}] ===")
        print(f"{'ep':>3} {vloss:>8} {vstd:>6} {'dead':>6} {'cos_same':>9} {'cos_diff':>9} {'margin':>8}")
        for _, r in v.iterrows():
            print(f"{int(r.epoch):>3} {r[vloss]:>8.4f} {r[vstd]:>6.3f} {r.dead_dim:>6.3f} "
                  f"{r.cos_same:>9.3f} {r.cos_diff:>9.3f} {r.margin:>8.3f}")


if __name__ == "__main__":
    main()
