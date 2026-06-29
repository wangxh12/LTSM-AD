"""Evaluation helpers for UAV anomaly detection."""

from .metrics import (
    binary_metrics,
    best_f1_threshold,
    collect_dataset_point_scores,
    collect_window_point_scores,
    compute_auprc,
    compute_auroc,
    reconstruct_dataset_points,
)
from .scoring import ReconstructionMahalanobisScorer

__all__ = [
    "binary_metrics",
    "best_f1_threshold",
    "collect_dataset_point_scores",
    "collect_window_point_scores",
    "compute_auprc",
    "compute_auroc",
    "reconstruct_dataset_points",
    "ReconstructionMahalanobisScorer",
]
