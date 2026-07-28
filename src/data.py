"""Shared preprocessing for Churn_Dataset.csv.

Loads the CSV once into RAM, imputes missing values with column medians,
standardizes features, and returns dense float32 train/test splits.
"""
import numpy as np
import pandas as pd


def load_churn(path: str, test_frac: float = 0.2, seed: int = 0):
    df = pd.read_csv(path)
    y = (df.pop("churned").astype(str).str.upper() == "TRUE").to_numpy(np.int32)
    X = df.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).to_numpy(np.float32)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    mu = X[train_idx].mean(axis=0)
    sd = X[train_idx].std(axis=0) + 1e-8
    X = (X - mu) / sd

    return (X[train_idx], y[train_idx]), (X[test_idx], y[test_idx])
