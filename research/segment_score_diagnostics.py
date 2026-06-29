#!/usr/bin/env python3
"""Inspect saved point anomaly scores by labeled anomaly segment."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from score_threshold_diagnostics import gmm_intersection


def load_scores(path: Path) -> tuple[list[float], list[int]]:
    scores: list[float] = []
    labels: list[int] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            score = float(row["score"])
            if not math.isfinite(score):
                continue
            scores.append(score)
            labels.append(int(float(row["label"]) > 0))
    if not scores:
        raise ValueError(f"No finite scores found in {path}")
    return scores, labels


def contiguous_segments(labels: list[int]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, label in enumerate(labels):
        if label and start is None:
            start = index
        if start is not None and (not label or index == len(labels) - 1):
            end = index - 1 if not label else index
            segments.append((start, end))
            start = None
    return segments


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def binary_metrics(labels: list[int], scores: list[float], threshold: float) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for label, score in zip(labels, scores):
        prediction = score >= threshold
        tp += int(prediction and label)
        fp += int(prediction and not label)
        fn += int(not prediction and label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def best_f1(labels: list[int], scores: list[float]) -> tuple[float, float, float, float]:
    paired = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    positives = sum(labels)
    tp = fp = 0
    fn = positives
    best_threshold = paired[0][0]
    best_precision = best_recall = best_score = 0.0
    index = 0
    while index < len(paired):
        threshold = paired[index][0]
        while index < len(paired) and paired[index][0] == threshold:
            label = paired[index][1]
            tp += int(label == 1)
            fp += int(label == 0)
            fn -= int(label == 1)
            index += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best_score:
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
            best_score = f1
    return best_threshold, best_precision, best_recall, best_score


def rolling(values: list[float], window: int, mode: str) -> list[float]:
    if window <= 1:
        return values[:]
    radius = window // 2
    output: list[float] = []
    for index in range(len(values)):
        left = max(0, index - radius)
        right = min(len(values), index + radius + 1)
        span = values[left:right]
        if mode == "mean":
            output.append(sum(span) / len(span))
        elif mode == "max":
            output.append(max(span))
        else:
            raise ValueError(f"Unknown rolling mode: {mode}")
    return output


def print_segment_table(labels: list[int], scores: list[float], threshold: float) -> None:
    print("segments")
    print("idx,start,end,len,recall@threshold,q05,q25,q50,q75,q95,max")
    for segment_index, (start, end) in enumerate(contiguous_segments(labels), start=1):
        values = scores[start : end + 1]
        detected = sum(score >= threshold for score in values)
        recall = detected / len(values)
        print(
            f"{segment_index},{start},{end},{len(values)},{recall:.4f},"
            f"{quantile(values, 0.05):.6f},{quantile(values, 0.25):.6f},"
            f"{quantile(values, 0.50):.6f},{quantile(values, 0.75):.6f},"
            f"{quantile(values, 0.95):.6f},{max(values):.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores_csv", type=Path)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    scores, labels = load_scores(args.scores_csv)
    normal_scores = [score for score, label in zip(scores, labels) if not label]
    anomaly_scores = [score for score, label in zip(scores, labels) if label]
    threshold = args.threshold if args.threshold is not None else quantile(normal_scores, 0.95)
    precision, recall, f1 = binary_metrics(labels, scores, threshold)
    best_threshold, best_precision, best_recall, best_score = best_f1(labels, scores)

    print(f"count normal={len(normal_scores)} anomaly={len(anomaly_scores)}")
    print(
        "normal q50/q90/q95/q99/max="
        f"{quantile(normal_scores, 0.50):.6f}/"
        f"{quantile(normal_scores, 0.90):.6f}/"
        f"{quantile(normal_scores, 0.95):.6f}/"
        f"{quantile(normal_scores, 0.99):.6f}/"
        f"{max(normal_scores):.6f}"
    )
    print(
        "anomaly q05/q25/q50/q75/q95/max="
        f"{quantile(anomaly_scores, 0.05):.6f}/"
        f"{quantile(anomaly_scores, 0.25):.6f}/"
        f"{quantile(anomaly_scores, 0.50):.6f}/"
        f"{quantile(anomaly_scores, 0.75):.6f}/"
        f"{quantile(anomaly_scores, 0.95):.6f}/"
        f"{max(anomaly_scores):.6f}"
    )
    print(
        f"threshold={threshold:.6f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}"
    )
    print(
        "best "
        f"threshold={best_threshold:.6f} precision={best_precision:.4f} "
        f"recall={best_recall:.4f} f1={best_score:.4f}"
    )
    gmm_threshold, weights, means, stds = gmm_intersection(scores)
    gmm_precision, gmm_recall, gmm_f1 = binary_metrics(labels, scores, gmm_threshold)
    print(
        f"gmm threshold={gmm_threshold:.6f} precision={gmm_precision:.4f} "
        f"recall={gmm_recall:.4f} f1={gmm_f1:.4f} "
        f"weights={[round(value, 4) for value in weights]} "
        f"means={[round(value, 4) for value in means]} "
        f"stds={[round(value, 4) for value in stds]}"
    )
    print_segment_table(labels, scores, threshold)

    print("rolling_best")
    print("mode,window,best_threshold,precision,recall,f1,gmm_threshold,gmm_precision,gmm_recall,gmm_f1")
    for mode in ("mean", "max"):
        for window in (3, 5, 9, 17, 33, 65, 129):
            smoothed = rolling(scores, window, mode)
            best_threshold, best_precision, best_recall, best_score = best_f1(labels, smoothed)
            gmm_threshold, _, _, _ = gmm_intersection(smoothed)
            gmm_precision, gmm_recall, gmm_f1 = binary_metrics(labels, smoothed, gmm_threshold)
            print(
                f"{mode},{window},{best_threshold:.6f},{best_precision:.4f},"
                f"{best_recall:.4f},{best_score:.4f},{gmm_threshold:.6f},"
                f"{gmm_precision:.4f},{gmm_recall:.4f},{gmm_f1:.4f}"
            )


if __name__ == "__main__":
    main()
