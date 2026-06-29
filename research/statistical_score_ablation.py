"""Compare strictly train-fitted statistical anomaly scores on a split-CSV dataset.

All transforms and score parameters are fit on ``data.train``.  The 95th
percentile threshold is fit on ``data.val``.  Test labels are used only for
reporting diagnostics.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.evaluation.metrics import best_f1_threshold, binary_metrics, compute_auprc, compute_auroc
from src.scripts.utils import load_config


def load_values(path: str, fields: list[str], label_col: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    values = frame[fields].apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both").bfill().ffill()
    labels = frame[label_col].to_numpy(dtype=np.int64) if label_col in frame else np.zeros(len(frame), dtype=np.int64)
    return values.to_numpy(dtype=np.float64), (labels > 0).astype(np.int64)


def fit_minmax(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = values.min(axis=0)
    scale = np.maximum(values.max(axis=0) - lower, 1e-6)
    return lower, scale


def z_scores(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.abs((values - mean) / scale)


def top_k_mean(values: np.ndarray, top_k: int) -> np.ndarray:
    return np.partition(values, -top_k, axis=1)[:, -top_k:].mean(axis=1)


def report(name: str, val_scores: np.ndarray, test_scores: np.ndarray, labels: np.ndarray) -> None:
    valid = np.isfinite(val_scores)
    threshold = float(np.percentile(val_scores[valid], 95))
    fixed = binary_metrics(labels, test_scores, threshold)
    best = best_f1_threshold(labels, test_scores)
    print(
        f"{name:>18} fixed_f1={fixed['f1']:.4f} p={fixed['precision']:.4f} r={fixed['recall']:.4f} "
        f"auroc={compute_auroc(labels, test_scores):.4f} auprc={compute_auprc(labels, test_scores):.4f} "
        f"best_f1={best['best_f1']:.4f} threshold={threshold:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    root = data["root_path"]
    fields = list(data["target_fields"])
    label_col = str(data.get("label_col", "label"))
    train, _ = load_values(f"{root}/{data['train']}", fields, label_col)
    val, _ = load_values(f"{root}/{data['val']}", fields, label_col)
    test, labels = load_values(f"{root}/{data['test']}", fields, label_col)

    lower, range_scale = fit_minmax(train)
    train_scaled = (train - lower) / range_scale
    val_scaled = (val - lower) / range_scale
    test_scaled = (test - lower) / range_scale
    mean = train_scaled.mean(axis=0)
    std = np.maximum(train_scaled.std(axis=0), 1e-6)
    val_z = z_scores(val_scaled, mean, std)
    test_z = z_scores(test_scaled, mean, std)

    report("diagonal_mean", val_z.mean(axis=1), test_z.mean(axis=1), labels)
    for top_k in (1, 2, 3, 4, 6, 9):
        report(f"diagonal_top{top_k}", top_k_mean(val_z, top_k), top_k_mean(test_z, top_k), labels)

    covariance = np.cov(train_scaled, rowvar=False)
    ridge = np.trace(covariance) / len(fields) * 1e-3
    precision = np.linalg.pinv(covariance + ridge * np.eye(len(fields)))
    val_centered = val_scaled - mean
    test_centered = test_scaled - mean
    report(
        "mahalanobis",
        np.sqrt(np.einsum("ni,ij,nj->n", val_centered, precision, val_centered)),
        np.sqrt(np.einsum("ni,ij,nj->n", test_centered, precision, test_centered)),
        labels,
    )

    _, eigenvectors = np.linalg.eigh(covariance)
    for components in (3, 6, 9, 12, 15):
        basis = eigenvectors[:, -components:]
        val_residual = val_centered - (val_centered @ basis) @ basis.T
        test_residual = test_centered - (test_centered @ basis) @ basis.T
        report(
            f"pca{components}_l1",
            np.abs(val_residual).mean(axis=1),
            np.abs(test_residual).mean(axis=1),
            labels,
        )

    train_delta = np.diff(train_scaled, axis=0)
    delta_mean = train_delta.mean(axis=0)
    delta_std = np.maximum(train_delta.std(axis=0), 1e-6)
    val_delta = z_scores(np.diff(val_scaled, axis=0), delta_mean, delta_std)
    test_delta = z_scores(np.diff(test_scaled, axis=0), delta_mean, delta_std)
    report("delta_mean", val_delta.mean(axis=1), test_delta.mean(axis=1), labels[1:])
    for top_k in (1, 3, 6):
        report(f"delta_top{top_k}", top_k_mean(val_delta, top_k), top_k_mean(test_delta, top_k), labels[1:])


if __name__ == "__main__":
    main()
