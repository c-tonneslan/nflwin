"""Train a win-probability model on nflfastR plays.

Two models for comparison:
  - logistic regression on raw features (the kind of baseline you'd actually
    sketch out at a whiteboard)
  - gradient-boosted trees, which handle the obvious non-linearities (e.g.
    score margin near zero in the 4th quarter dominates everything)

Split is by season: train 2018-2022, hold out 2023. No play from a test game
ever leaks into training.
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURES = [
    "down",
    "ydstogo",
    "yardline_100",
    "score_differential",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "qtr",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
]


def evaluate(name, y_true, y_prob):
    return {
        "model": name,
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }


def baseline_prob(df):
    """Score-only baseline: a logistic on score_differential alone."""
    return df["score_differential"].apply(lambda d: 1 / (1 + np.exp(-0.16 * d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/plays.parquet")
    ap.add_argument("--test-season", type=int, default=2023)
    ap.add_argument("--val-season", type=int, default=2022,
                    help="season used as validation for xgb early stopping")
    ap.add_argument("--out-dir", default="models")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    print(f"loaded {len(df):,} plays from {df['season'].min()}-{df['season'].max()}")

    train = df[~df["season"].isin([args.val_season, args.test_season])].reset_index(drop=True)
    val = df[df["season"] == args.val_season].reset_index(drop=True)
    test = df[df["season"] == args.test_season].reset_index(drop=True)
    print(f"  train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")

    Xtr, ytr = train[FEATURES].values, train["posteam_won"].values
    Xva, yva = val[FEATURES].values, val["posteam_won"].values
    Xte, yte = test[FEATURES].values, test["posteam_won"].values

    os.makedirs(args.out_dir, exist_ok=True)
    results = []

    # naive: 0.5 everywhere
    results.append(evaluate("constant_0.5", yte, np.full_like(yte, 0.5, dtype=float)))

    # score-only baseline
    p_score = baseline_prob(test).values
    results.append(evaluate("score_only_logit", yte, p_score))

    # logistic regression with all features
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    logit = LogisticRegression(max_iter=2000, C=1.0)
    logit.fit(Xtr_s, ytr)
    p_logit = logit.predict_proba(Xte_s)[:, 1]
    results.append(evaluate("logistic", yte, p_logit))

    # gradient boosted trees with early stopping on a held-out season
    xgb = XGBClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        early_stopping_rounds=25,
        n_jobs=-1,
    )
    xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    print(f"xgb stopped at iteration {xgb.best_iteration} (val logloss {xgb.best_score:.4f})")
    p_xgb = xgb.predict_proba(Xte)[:, 1]
    results.append(evaluate("xgboost", yte, p_xgb))

    print()
    print(f"{'model':<22} {'log_loss':>10} {'brier':>10} {'auc':>8}")
    for r in results:
        print(f"{r['model']:<22} {r['log_loss']:>10.4f} {r['brier']:>10.4f} {r['auc']:>8.4f}")

    # save artifacts
    joblib.dump({"scaler": scaler, "model": logit, "features": FEATURES},
                os.path.join(args.out_dir, "logistic.joblib"))
    xgb.save_model(os.path.join(args.out_dir, "xgboost.json"))

    # feature importances for the writeup
    importances = sorted(
        zip(FEATURES, xgb.feature_importances_), key=lambda kv: -kv[1]
    )
    print("\nxgb feature importance (gain-normalized):")
    for name, imp in importances:
        print(f"  {name:<32} {imp:.3f}")

    # calibration data for charts
    pt_xgb, pp_xgb = calibration_curve(yte, p_xgb, n_bins=20, strategy="quantile")
    pt_log, pp_log = calibration_curve(yte, p_logit, n_bins=20, strategy="quantile")
    pt_score, pp_score = calibration_curve(yte, p_score, n_bins=20, strategy="quantile")
    cal = {
        "xgb": {"pred": pp_xgb.tolist(), "true": pt_xgb.tolist()},
        "logit": {"pred": pp_log.tolist(), "true": pt_log.tolist()},
        "score_only": {"pred": pp_score.tolist(), "true": pt_score.tolist()},
    }

    out = {
        "metrics": results,
        "feature_importance": [{"feature": k, "importance": float(v)} for k, v in importances],
        "calibration": cal,
        "test_season": args.test_season,
        "n_train": len(train),
        "n_test": len(test),
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(out, f, indent=2)

    # predictions on the test set for charts
    test_out = test[["game_id", "qtr", "game_seconds_remaining", "posteam",
                     "defteam", "home_team", "away_team", "score_differential",
                     "posteam_won"]].copy()
    test_out["wp_xgb"] = p_xgb
    test_out["wp_logit"] = p_logit
    test_out.to_parquet(os.path.join(args.out_dir, "test_predictions.parquet"))


if __name__ == "__main__":
    main()
