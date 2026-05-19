"""Quick CLI for scoring a single game state.

  $ python scripts/predict.py --down 3 --ydstogo 8 --yardline 65 \\
      --margin -7 --seconds-left 480 --qtr 4 --to-off 2 --to-def 3
  wp = 0.213  (xgb)
"""

import argparse

import numpy as np
import xgboost as xgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--down", type=int, required=True)
    ap.add_argument("--ydstogo", type=int, required=True)
    ap.add_argument("--yardline", type=int, required=True,
                    help="yards from offense to opposing endzone (1-99)")
    ap.add_argument("--margin", type=int, required=True,
                    help="score differential, offense - defense")
    ap.add_argument("--seconds-left", type=int, required=True,
                    help="seconds remaining in the game")
    ap.add_argument("--qtr", type=int, required=True)
    ap.add_argument("--to-off", type=int, default=3)
    ap.add_argument("--to-def", type=int, default=3)
    ap.add_argument("--model", default="models/xgboost.json")
    args = ap.parse_args()

    if not 1 <= args.qtr <= 5:
        raise SystemExit("qtr must be 1-5 (5 is overtime)")
    if not 1 <= args.down <= 4:
        raise SystemExit("down must be 1-4")
    if not 1 <= args.yardline <= 99:
        raise SystemExit("yardline must be 1-99")

    if args.qtr <= 2:
        half = max(0, args.seconds_left - 1800)
    else:
        half = args.seconds_left

    row = np.array([[
        args.down, args.ydstogo, args.yardline, args.margin,
        args.seconds_left, half, args.qtr, args.to_off, args.to_def,
    ]], dtype=float)

    clf = xgb.XGBClassifier()
    clf.load_model(args.model)
    p = clf.predict_proba(row)[0, 1]
    print(f"wp = {p:.3f}  (xgb)")


if __name__ == "__main__":
    main()
