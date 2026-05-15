"""Render charts for the writeup.

Outputs go to charts/:
  - calibration.png  : predicted vs empirical WP across deciles
  - importance.png   : feature importance bar
  - margin_grid.png  : WP curves by score margin, broken out by quarter
  - sb_lviii.png     : Super Bowl LVIII (49ers/Chiefs) WP over time

Matplotlib only, no theming libraries.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})


def calibration_chart(report, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect calibration")
    for label, key, color in [
        ("xgboost", "xgb", "#0a84ff"),
        ("logistic", "logit", "#ff8c00"),
        ("score only", "score_only", "#888888"),
    ]:
        c = report["calibration"][key]
        ax.plot(c["pred"], c["true"], marker="o", ms=4, lw=1.5, color=color, label=label)
    ax.set_xlabel("predicted win probability")
    ax.set_ylabel("empirical win rate")
    ax.set_title("calibration on held-out 2023 plays")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"  wrote {out_path}")


def importance_chart(report, out_path):
    fi = report["feature_importance"]
    names = [f["feature"] for f in fi][::-1]
    vals = [f["importance"] for f in fi][::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(names, vals, color="#0a84ff", alpha=0.85)
    ax.set_xlabel("gain-normalized importance")
    ax.set_title("xgboost feature importance")
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=9, color="#444")
    ax.set_xlim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"  wrote {out_path}")


def margin_grid(plays, out_path):
    """Average WP at margin m, broken out by quarter."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True, sharex=True)
    for q, ax in zip([1, 2, 3, 4], axes.flat):
        sub = plays[plays["qtr"] == q]
        bins = np.arange(-21, 22, 2)
        sub = sub.assign(margin_bin=pd.cut(sub["score_differential"], bins))
        agg = sub.groupby("margin_bin", observed=True)["wp_xgb"].mean()
        centers = [(b.left + b.right) / 2 for b in agg.index]
        ax.plot(centers, agg.values, marker="o", ms=4, color="#0a84ff")
        ax.set_title(f"Q{q}")
        ax.set_xlabel("score margin (offense - defense)")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="#999", lw=0.6, ls="--")
        ax.axvline(0, color="#999", lw=0.6, ls="--")
    fig.suptitle("avg model WP vs score margin, by quarter", y=1.0)
    fig.supylabel("win probability")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"  wrote {out_path}")


def superbowl_chart(plays, game_id, out_path):
    g = plays[plays["game_id"] == game_id].copy()
    if g.empty:
        print(f"  skipped {game_id} (not in test set)")
        return
    # express WP from the home team's POV
    g["wp_home"] = np.where(g["posteam"] == g["home_team"], g["wp_xgb"], 1 - g["wp_xgb"])
    minutes = (3600 - g["game_seconds_remaining"]) / 60

    home, away = g["home_team"].iloc[0], g["away_team"].iloc[0]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.fill_between(minutes, 0.5, g["wp_home"], where=g["wp_home"] >= 0.5,
                    color="#d62728", alpha=0.6, label=home)
    ax.fill_between(minutes, 0.5, g["wp_home"], where=g["wp_home"] < 0.5,
                    color="#0a84ff", alpha=0.6, label=away)
    ax.plot(minutes, g["wp_home"], color="#222", lw=0.8)
    for q in [15, 30, 45, 60]:
        ax.axvline(q, color="#999", lw=0.5)
    ax.axhline(0.5, color="#444", lw=0.6, ls="--")
    ax.set_xlim(0, minutes.max() + 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("game minutes elapsed")
    ax.set_ylabel(f"{home} win probability")
    ax.set_title(f"{away} @ {home}  ({game_id})")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"  wrote {out_path}")


def main():
    os.makedirs("charts", exist_ok=True)
    with open("models/report.json") as f:
        report = json.load(f)
    plays = pd.read_parquet("models/test_predictions.parquet")
    calibration_chart(report, "charts/calibration.png")
    importance_chart(report, "charts/importance.png")
    margin_grid(plays, "charts/margin_grid.png")
    superbowl_chart(plays, "2023_22_SF_KC", "charts/sb_lviii.png")


if __name__ == "__main__":
    main()
