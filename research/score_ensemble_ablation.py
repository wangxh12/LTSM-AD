"""Test a validation-normalized v1 reconstruction + Mahalanobis score ensemble."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import (
    best_f1_threshold,
    binary_metrics,
    compute_auprc,
    compute_auroc,
    reconstruct_dataset_points,
)
from src.scripts import finetune
from src.scripts.utils import device_from_trainer_config, load_config
from research.score_threshold_diagnostics import gmm_intersection


def read_values(path: str, fields: list[str], label_col: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    values = frame[fields].apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both").bfill().ffill()
    labels = frame[label_col].to_numpy(dtype=np.int64)
    return values.to_numpy(dtype=np.float64), (labels > 0).astype(np.int64)


def loader(dataset, batch_size: int, num_workers: int) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def normalize_from_validation(values: np.ndarray, validation_values: np.ndarray) -> np.ndarray:
    median = float(np.median(validation_values))
    mad = float(np.median(np.abs(validation_values - median)))
    if mad < 1e-6:
        raise ValueError("Validation score MAD is zero; cannot normalize score components")
    return (values - median) / mad


def report(
    name: str,
    validation_scores: np.ndarray,
    test_scores: np.ndarray,
    labels: np.ndarray,
    threshold_percentile: float,
) -> None:
    threshold = float(np.percentile(validation_scores, threshold_percentile))
    fixed = binary_metrics(labels, test_scores, threshold)
    best = best_f1_threshold(labels, test_scores)
    print(
        f"{name:>14} fixed_f1={fixed['f1']:.4f} p={fixed['precision']:.4f} r={fixed['recall']:.4f} "
        f"auroc={compute_auroc(labels, test_scores):.4f} auprc={compute_auprc(labels, test_scores):.4f} "
        f"best_f1={best['best_f1']:.4f}"
    )
    gmm_threshold, _, _, _ = gmm_intersection(test_scores.tolist())
    gmm = binary_metrics(labels, test_scores, gmm_threshold)
    print(
        f"{'transductive_gmm':>14} f1={gmm['f1']:.4f} p={gmm['precision']:.4f} "
        f"r={gmm['recall']:.4f} threshold={gmm_threshold:.6f}"
    )


def mahalanobis_scores(train: np.ndarray, values: np.ndarray) -> np.ndarray:
    lower = train.min(axis=0)
    scale = np.maximum(train.max(axis=0) - lower, 1e-6)
    train_scaled = (train - lower) / scale
    values_scaled = (values - lower) / scale
    mean = train_scaled.mean(axis=0)
    covariance = np.cov(train_scaled, rowvar=False)
    ridge = np.trace(covariance) / covariance.shape[0] * 1e-3
    precision = np.linalg.pinv(covariance + ridge * np.eye(covariance.shape[0]))
    centered = values_scaled - mean
    return np.sqrt(np.einsum("ni,ij,nj->n", centered, precision, centered))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    root = data["root_path"]
    fields = list(data["target_fields"])
    label_col = str(data.get("label_col", "label"))
    train_values, _ = read_values(f"{root}/{data['train']}", fields, label_col)
    val_values, _ = read_values(f"{root}/{data['val']}", fields, label_col)
    test_values, labels = read_values(f"{root}/{data['test']}", fields, label_col)

    datamodule = finetune.get_datamodule(config, setup=True)
    device = device_from_trainer_config(config)
    model = finetune.load_finetuned_module(
        config.get("pretrained_model"), args.checkpoint, config, map_location=device
    ).to(device)
    loader_kwargs = {"batch_size": int(data.get("batch_size", 256)), "num_workers": int(data.get("num_workers", 0))}
    if datamodule.threshold_dataset is None:
        raise RuntimeError("DataModule did not provide a threshold dataset")
    val_outputs = reconstruct_dataset_points(
        model, loader(datamodule.threshold_dataset, **loader_kwargs), datamodule.threshold_dataset, device
    )
    if len(datamodule.test_files) != 1:
        raise ValueError("This diagnostic requires exactly one configured test file")
    test_dataset = datamodule.make_test_dataset(datamodule.test_files[0])
    test_outputs = reconstruct_dataset_points(model, loader(test_dataset, **loader_kwargs), test_dataset, device)

    val_reconstruction = val_outputs["scores"]
    test_reconstruction = test_outputs["scores"]
    val_mahalanobis = mahalanobis_scores(train_values, val_values)
    test_mahalanobis = mahalanobis_scores(train_values, test_values)
    print(f"threshold_percentile={args.threshold_percentile:.1f}")
    report("v1", val_reconstruction, test_reconstruction, labels, args.threshold_percentile)
    report("mahalanobis", val_mahalanobis, test_mahalanobis, labels, args.threshold_percentile)

    normalized_v1_val = normalize_from_validation(val_reconstruction, val_reconstruction)
    normalized_v1_test = normalize_from_validation(test_reconstruction, val_reconstruction)
    normalized_mahal_val = normalize_from_validation(val_mahalanobis, val_mahalanobis)
    normalized_mahal_test = normalize_from_validation(test_mahalanobis, val_mahalanobis)
    for reconstruction_weight in (0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0):
        report(
            f"blend_{reconstruction_weight:.2f}",
            reconstruction_weight * normalized_v1_val + (1.0 - reconstruction_weight) * normalized_mahal_val,
            reconstruction_weight * normalized_v1_test + (1.0 - reconstruction_weight) * normalized_mahal_test,
            labels,
            args.threshold_percentile,
        )
    report(
        "max_pair",
        np.maximum(normalized_v1_val, normalized_mahal_val),
        np.maximum(normalized_v1_test, normalized_mahal_test),
        labels,
        args.threshold_percentile,
    )
    report(
        "min_pair",
        np.minimum(normalized_v1_val, normalized_mahal_val),
        np.minimum(normalized_v1_test, normalized_mahal_test),
        labels,
        args.threshold_percentile,
    )


if __name__ == "__main__":
    main()
