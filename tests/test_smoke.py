"""Smoke tests. These run after `make all` has produced the artifacts."""

import json
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _needs(path):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        pytest.skip(f"missing {path} (run `make all`)")
    return full


def test_plays_parquet():
    df = pd.read_parquet(_needs("data/plays.parquet"))
    assert len(df) > 100_000
    assert df["posteam_won"].isin([0, 1]).all()
    assert 0.4 < df["posteam_won"].mean() < 0.6


def test_xgb_predicts_in_range():
    model_path = _needs("models/xgboost.json")
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
    # plausible 4Q down-by-7 state
    row = np.array([[3, 8, 65, -7, 480, 480, 4, 2, 3]], dtype=float)
    p = clf.predict_proba(row)[0, 1]
    assert 0 < p < 0.5  # losing late, should be under 50%


def test_xgb_monotone_in_margin():
    """Holding everything else fixed, a bigger lead should never lower WP much."""
    clf = xgb.XGBClassifier()
    clf.load_model(_needs("models/xgboost.json"))
    base = [1, 10, 50, 0, 1800, 1800, 2, 3, 3]
    probs = []
    for margin in range(-14, 15, 2):
        row = base.copy()
        row[3] = margin
        probs.append(clf.predict_proba(np.array([row], dtype=float))[0, 1])
    # not strictly monotone (gbt can be noisy) but the trend has to be there
    assert probs[-1] > probs[0]
    assert probs[-1] - probs[0] > 0.4


def test_logistic_roundtrip():
    bundle = joblib.load(_needs("models/logistic.joblib"))
    assert "scaler" in bundle and "model" in bundle
    assert len(bundle["features"]) == 9


def test_report_has_metrics():
    with open(_needs("models/report.json")) as f:
        rep = json.load(f)
    models = {m["model"] for m in rep["metrics"]}
    assert "xgboost" in models and "logistic" in models
    xgb_row = next(m for m in rep["metrics"] if m["model"] == "xgboost")
    assert xgb_row["log_loss"] < 0.6
    assert xgb_row["auc"] > 0.8


def test_predict_cli_rejects_bad_qtr():
    """The CLI should refuse a quarter outside 1-5 instead of returning a number."""
    script = os.path.join(ROOT, "scripts", "predict.py")
    result = subprocess.run(
        [
            sys.executable, script,
            "--down", "1", "--ydstogo", "10", "--yardline", "50",
            "--margin", "0", "--seconds-left", "1800", "--qtr", "9",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "qtr" in result.stderr.lower()
