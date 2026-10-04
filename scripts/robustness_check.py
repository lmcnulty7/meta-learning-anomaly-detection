"""Robustness checks behind the "Results" section of README.md.

Uses the committed performance_matrix.csv (the F1 matrix behind the paper) and
meta-features recomputed from data/nab_cleaned/, then prints:
  - the majority-class and uniform-random baselines,
  - leave-one-dataset-out accuracy with the paper's Random Forest seed (42),
  - mean / min / max over 20 Random Forest seeds (0-19),
  - the same two numbers without the label-derived anomaly_ratio feature.

Usage (from the repository root, after scripts/clean_nab.py; takes a few minutes):
  python scripts/robustness_check.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import meta_learner as ml  # noqa: E402

F1_COLS = ["iso_f1", "pca_f1", "ocsvm_f1", "kmeans_f1", "ae_f1"]
META_COLS = ["mean", "std", "skew", "kurtosis", "acf1", "adf_stat", "adf_p", "anomaly_ratio", "length"]


def meta_features() -> pd.DataFrame:
    with open(ml.LABELS_PATH) as f:
        labels = json.load(f)
    rows = {}
    for path in ml.DATA_DIR.glob("**/*.csv"):  # same file order as meta_learner.py
        key = str(path.relative_to(ml.DATA_DIR).with_suffix("")).replace("\\", "/")
        rows[key] = ml.compute_meta_features(ml.apply_labels(pd.read_csv(path), key, labels))
    return pd.DataFrame(rows).T


def loocv_accuracy(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced")
    pred = cross_val_predict(rf, X, y, cv=LeaveOneOut(), n_jobs=-1)
    return float((pred == y).mean())


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # constant series (art_flatline)
    meta = meta_features()
    if meta.empty:
        raise SystemExit("No cleaned CSVs in data/nab_cleaned/. Run scripts/clean_nab.py first.")
    perf = pd.read_csv(REPO_ROOT / "performance_matrix.csv", index_col="dataset")
    combined = meta.join(perf, how="inner")

    # Same target as meta_learner.py: idxmax keeps the first column on ties.
    y = combined[F1_COLS].idxmax(axis=1).map({c: i for i, c in enumerate(F1_COLS)}).to_numpy()
    counts = np.bincount(y, minlength=len(F1_COLS))
    print(f"series: {len(y)}; best-detector counts: {dict(zip(F1_COLS, counts.tolist()))}")
    print(f"majority-class baseline (always {F1_COLS[counts.argmax()]}): {counts.max() / len(y):.1%}")
    print(f"uniform random baseline: {1 / len(F1_COLS):.1%}")

    feature_sets = [
        ("all 9 meta-features", META_COLS),
        ("without anomaly_ratio", [c for c in META_COLS if c != "anomaly_ratio"]),
    ]
    for name, cols in feature_sets:
        X = combined[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        accs = np.array([loocv_accuracy(X, y, seed) for seed in range(20)])
        print(
            f"{name}: seed 42 = {loocv_accuracy(X, y, 42):.1%}; "
            f"seeds 0-19 mean = {accs.mean():.1%} (range {accs.min():.1%} to {accs.max():.1%})"
        )


if __name__ == "__main__":
    main()
