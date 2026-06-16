from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _time_axis(time: np.ndarray | None, length: int) -> np.ndarray:
    if time is None or len(time) != length:
        return np.arange(length)
    return time


def _anomaly_spans(labels: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    in_span = False
    start = 0
    for index, value in enumerate(labels.astype(bool)):
        if value and not in_span:
            start = index
            in_span = True
        elif not value and in_span:
            spans.append((start, index - 1))
            in_span = False
    if in_span:
        spans.append((start, len(labels) - 1))
    return spans


def _shade_anomalies(ax: plt.Axes, axis: np.ndarray, labels: np.ndarray) -> None:
    for start, end in _anomaly_spans(labels):
        ax.axvspan(axis[start], axis[end], color="tab:red", alpha=0.12, linewidth=0)


def plot_reconstruction(
    output_dir: str | Path,
    dataset_name: str,
    feature_names: list[str],
    target: np.ndarray,
    reconstruction: np.ndarray,
    labels: np.ndarray,
    time: np.ndarray | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    axis = _time_axis(time, len(target))

    for feature_index, feature_name in enumerate(feature_names):
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(axis, target[:, feature_index], label="ground truth", linewidth=1.2)
        ax.plot(axis, reconstruction[:, feature_index], label="reconstruction", linewidth=1.0, alpha=0.85)
        _shade_anomalies(ax, axis, labels)
        ax.set_title(f"{dataset_name} - {feature_name}")
        ax.set_xlabel("time")
        ax.set_ylabel(feature_name)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"{dataset_name}_{feature_name}_reconstruction.png", dpi=160)
        plt.close(fig)


def plot_scores(
    output_path: str | Path,
    dataset_name: str,
    scores: np.ndarray,
    threshold: float,
    labels: np.ndarray,
    best_threshold: float | None = None,
    time: np.ndarray | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    axis = _time_axis(time, len(scores))

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(axis, scores, label="point MAE score", linewidth=1.2, color="tab:blue")
    ax.axhline(threshold, label=f"fixed threshold={threshold:.6g}", color="tab:orange", linewidth=1.2)
    if best_threshold is not None and np.isfinite(best_threshold):
        ax.axhline(best_threshold, label=f"best-F1 threshold={best_threshold:.6g}", color="tab:green", linewidth=1.0)
    _shade_anomalies(ax, axis, labels)
    ax.set_title(f"{dataset_name} - anomaly score")
    ax.set_xlabel("time")
    ax.set_ylabel("MAE")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

