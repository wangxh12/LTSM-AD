from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data import SequentialSampler

from src.data.csv_windows import FlightDataset


def _as_numpy(values: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return values


@torch.no_grad()
def collect_window_point_scores(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Collect flattened point MAE scores from all windows in a loader."""

    model.eval()
    scores: list[np.ndarray] = []
    for batch in loader:
        x = batch.series.to(device)
        reconstruction = model(x)
        batch_scores = (reconstruction - x).abs().mean(dim=-1)
        scores.append(_as_numpy(batch_scores).reshape(-1))
    if not scores:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(scores).astype(np.float32)


@torch.no_grad()
def collect_dataset_point_scores(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    """Return scores using the same overlap averaging applied to test data."""

    if isinstance(dataset, FlightDataset):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        return reconstruct_dataset_points(model, loader, dataset, device)["scores"]
    if isinstance(dataset, ConcatDataset):
        return np.concatenate(
            [
                collect_dataset_point_scores(model, child, device, batch_size, num_workers)
                for child in dataset.datasets
            ]
        )
    raise TypeError(f"Point-score reconstruction requires WindowDataset or ConcatDataset, got {type(dataset)!r}")


@torch.no_grad()
def reconstruct_dataset_points(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: FlightDataset,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Average overlapping window reconstructions and scores back to point level."""

    model.eval()
    series_length, num_features = dataset.series.shape
    recon_sum = np.zeros((series_length, num_features), dtype=np.float64)
    feature_score_sum = np.zeros((series_length, num_features), dtype=np.float64)
    score_sum = np.zeros(series_length, dtype=np.float64)
    counts = np.zeros(series_length, dtype=np.float64)

    if not isinstance(loader.sampler, SequentialSampler):
        raise ValueError("Point reconstruction requires a DataLoader with sequential sampling")

    window_offset = 0
    for batch in loader:
        x = batch.series.to(device)
        reconstruction = model(x)
        feature_scores = (reconstruction - x).abs()
        scores = feature_scores.mean(dim=-1)
        reconstruction_np = _as_numpy(reconstruction)
        feature_scores_np = _as_numpy(feature_scores)
        scores_np = _as_numpy(scores)
        batch_size = x.shape[0]
        starts = dataset.starts[window_offset : window_offset + batch_size]
        window_offset += batch_size

        for row, start in enumerate(starts):
            end = start + dataset.seq_len
            recon_sum[start:end] += reconstruction_np[row]
            feature_score_sum[start:end] += feature_scores_np[row]
            score_sum[start:end] += scores_np[row]
            counts[start:end] += 1.0

    if window_offset != len(dataset):
        raise ValueError(
            f"DataLoader yielded {window_offset} windows, but the dataset contains {len(dataset)}"
        )

    valid = counts > 0
    reconstruction = np.full_like(recon_sum, np.nan, dtype=np.float32)
    point_feature_scores = np.full_like(feature_score_sum, np.nan, dtype=np.float32)
    point_scores = np.full(series_length, np.nan, dtype=np.float32)
    reconstruction[valid] = (recon_sum[valid] / counts[valid, None]).astype(np.float32)
    point_feature_scores[valid] = (feature_score_sum[valid] / counts[valid, None]).astype(np.float32)
    point_scores[valid] = (score_sum[valid] / counts[valid]).astype(np.float32)

    labels = dataset.labels if dataset.labels is not None else np.zeros(series_length, dtype=np.int64)
    return {
        "target": dataset.series.astype(np.float32),
        "reconstruction": reconstruction,
        "feature_scores": point_feature_scores,
        "scores": point_scores,
        "labels": labels.astype(np.int64),
        "valid": valid,
    }


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = labels.astype(np.int64)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    predictions = (scores >= threshold).astype(np.int64)

    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = labels.astype(np.int64)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    if labels.sum() == 0 or len(labels) == 0:
        return {
            "best_f1": 0.0,
            "best_threshold": float(np.nan),
            "best_tp": 0,
            "best_fp": 0,
            "best_tn": int((labels == 0).sum()),
            "best_fn": int((labels == 1).sum()),
            "best_accuracy": 0.0,
            "best_precision": 0.0,
            "best_recall": 0.0,
        }

    order = np.argsort(-scores)
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    positives = max(int((labels == 1).sum()), 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    f1 = np.where(
        precision + recall > 0,
        2.0 * precision * recall / np.maximum(precision + recall, 1e-12),
        0.0,
    )
    best_index = int(np.nanargmax(f1))
    best_threshold = float(sorted_scores[best_index])
    best_metrics = binary_metrics(labels, scores, best_threshold)
    return {
        "best_f1": float(f1[best_index]),
        "best_threshold": best_threshold,
        "best_tp": best_metrics["tp"],
        "best_fp": best_metrics["fp"],
        "best_tn": best_metrics["tn"],
        "best_fn": best_metrics["fn"],
        "best_accuracy": best_metrics["accuracy"],
        "best_precision": best_metrics["precision"],
        "best_recall": best_metrics["recall"],
    }


def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives))


def compute_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    positives = int((labels == 1).sum())
    if positives == 0 or len(labels) == 0:
        return float("nan")

    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    recall_delta = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(recall_delta * precision))
