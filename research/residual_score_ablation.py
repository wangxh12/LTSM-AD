"""Evaluate validation-calibrated feature residual aggregations for one checkpoint.

The script fits every calibration statistic on the configured threshold split only.
Test labels are read solely to report diagnostic metrics, never to fit a scorer or
choose a threshold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.evaluation.metrics import (
    best_f1_threshold,
    binary_metrics,
    reconstruct_dataset_points,
)
from src.scripts import finetune
from src.scripts.utils import device_from_trainer_config, load_config


def make_loader(dataset, batch_size: int, num_workers: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def robust_feature_scale(feature_scores: np.ndarray) -> np.ndarray:
    median = np.nanmedian(feature_scores, axis=0)
    median = np.maximum(median, 1e-6)
    return median.astype(np.float32)


def aggregate(feature_scores: np.ndarray, scale: np.ndarray, top_k: int | None) -> np.ndarray:
    normalized = feature_scores / scale[None, :]
    if top_k is None:
        return normalized.mean(axis=1)
    return np.partition(normalized, -top_k, axis=1)[:, -top_k:].mean(axis=1)


def report(name: str, threshold_scores: np.ndarray, test_scores: np.ndarray, labels: np.ndarray) -> None:
    valid_threshold_scores = threshold_scores[np.isfinite(threshold_scores)]
    fixed_threshold = float(np.percentile(valid_threshold_scores, 95))
    fixed = binary_metrics(labels, test_scores, fixed_threshold)
    best = best_f1_threshold(labels, test_scores)
    print(
        f"{name:>16} fixed_f1={fixed['f1']:.4f} "
        f"precision={fixed['precision']:.4f} recall={fixed['recall']:.4f} "
        f"threshold={fixed_threshold:.6f} best_f1={best['best_f1']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    datamodule = finetune.get_datamodule(config, setup=True)
    device = device_from_trainer_config(config)
    model = finetune.load_finetuned_module(
        config.get("pretrained_model"),
        args.checkpoint,
        config,
        map_location=device,
    ).to(device)

    data_cfg = config["data"]
    loader_kwargs = {
        "batch_size": int(data_cfg.get("batch_size", 256)),
        "num_workers": int(data_cfg.get("num_workers", 0)),
    }
    threshold_dataset = datamodule.threshold_dataset
    if threshold_dataset is None:
        raise RuntimeError("DataModule did not provide a threshold dataset")
    threshold_outputs = reconstruct_dataset_points(
        model,
        make_loader(threshold_dataset, **loader_kwargs),
        threshold_dataset,
        device,
    )
    if len(datamodule.test_files) != 1:
        raise ValueError("This diagnostic requires exactly one configured test file")
    test_dataset = datamodule.make_test_dataset(datamodule.test_files[0])
    test_outputs = reconstruct_dataset_points(
        model,
        make_loader(test_dataset, **loader_kwargs),
        test_dataset,
        device,
    )
    labels = test_outputs["labels"]
    if labels.sum() == 0:
        raise ValueError("Test set has no anomaly labels; diagnostic F1 is undefined")

    threshold_features = threshold_outputs["feature_scores"]
    test_features = test_outputs["feature_scores"]
    scale = robust_feature_scale(threshold_features)
    print("feature median MAE scales:", np.array2string(scale, precision=5))

    for top_k in (None, 1, 2, 3, 4, 6, 9):
        name = "mean" if top_k is None else f"top{top_k}"
        report(
            name,
            aggregate(threshold_features, scale, top_k),
            aggregate(test_features, scale, top_k),
            labels,
        )


if __name__ == "__main__":
    main()
