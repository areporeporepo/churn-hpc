"""Shared preprocessing for Churn_Dataset.csv.

Loads the CSV once into RAM, imputes missing values with column medians,
standardizes features, and returns dense float32 train/test splits.

Falls back to a stdlib-csv + numpy parser when pandas is absent, so the
pinned public pytorch/pytorch image runs it with zero runtime installs.
"""
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None


def _load_numpy(path):
    import csv
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    label_i = header.index("churned")
    y = np.array([r[label_i].strip().upper() == "TRUE" for r in rows], np.int32)
    feats = [[float(v) if v.strip() else np.nan
              for j, v in enumerate(r) if j != label_i] for r in rows]
    return np.array(feats, np.float32), y


def load_churn(path: str, test_frac: float = 0.2, seed: int = 0):
    if pd is not None:
        df = pd.read_csv(path)
        y = (df.pop("churned").astype(str).str.upper() == "TRUE").to_numpy(np.int32)
        X = df.apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    else:
        X, y = _load_numpy(path)
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), med, X)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    mu = X[train_idx].mean(axis=0)
    sd = X[train_idx].std(axis=0) + 1e-8
    X = (X - mu) / sd

    return (X[train_idx], y[train_idx]), (X[test_idx], y[test_idx])
