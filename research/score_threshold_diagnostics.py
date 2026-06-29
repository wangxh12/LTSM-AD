#!/usr/bin/env python3
"""Compare label-free score threshold rules on a saved anomaly-score CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import median, pstdev


def load_scores(path: Path) -> list[tuple[float, int]]:
    pairs: list[tuple[float, int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            score = float(row["score"])
            if math.isfinite(score):
                pairs.append((score, int(float(row["label"]) > 0)))
    if not pairs:
        raise ValueError(f"No finite scores found in {path}")
    return pairs


def quantile(values: list[float], q: float) -> float:
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def metrics(pairs: list[tuple[float, int]], threshold: float) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for score, label in pairs:
        prediction = score >= threshold
        tp += int(prediction and label)
        fp += int(prediction and not label)
        fn += int(not prediction and label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def otsu_threshold(values: list[float], bins: int = 256) -> float:
    lower, upper = min(values), max(values)
    width = (upper - lower) / bins
    if width == 0:
        return lower
    counts = [0] * bins
    for value in values:
        counts[min(int((value - lower) / width), bins - 1)] += 1

    total = len(values)
    total_sum = sum(index * count for index, count in enumerate(counts))
    class0_count = class0_sum = 0.0
    best_variance = -1.0
    best_index = 0
    for index, count in enumerate(counts[:-1]):
        class0_count += count
        if class0_count == 0:
            continue
        class1_count = total - class0_count
        if class1_count == 0:
            break
        class0_sum += index * count
        class0_mean = class0_sum / class0_count
        class1_mean = (total_sum - class0_sum) / class1_count
        variance = class0_count * class1_count * (class0_mean - class1_mean) ** 2
        if variance > best_variance:
            best_variance = variance
            best_index = index
    return lower + (best_index + 0.5) * width


def normal_pdf(value: float, mean: float, std: float) -> float:
    std = max(std, 1e-8)
    z = (value - mean) / std
    return math.exp(-0.5 * z * z) / (std * math.sqrt(2 * math.pi))


def gmm_intersection(values: list[float], steps: int = 100) -> tuple[float, list[float], list[float], list[float]]:
    ordered = sorted(values)
    means = [ordered[len(values) // 4], ordered[3 * len(values) // 4]]
    initial_std = max(pstdev(values), 1e-5)
    stds = [initial_std, initial_std]
    weights = [0.5, 0.5]

    for _ in range(steps):
        responsibilities: list[float] = []
        for value in values:
            left = weights[0] * normal_pdf(value, means[0], stds[0])
            right = weights[1] * normal_pdf(value, means[1], stds[1])
            responsibilities.append(left / (left + right) if left + right else 0.5)
        for component, response in enumerate((responsibilities, [1.0 - x for x in responsibilities])):
            total = sum(response)
            weights[component] = total / len(values)
            means[component] = sum(weight * value for weight, value in zip(response, values)) / max(total, 1e-8)
            variance = sum(
                weight * (value - means[component]) ** 2 for weight, value in zip(response, values)
            ) / max(total, 1e-8)
            stds[component] = max(math.sqrt(variance), 1e-6)

    if means[0] > means[1]:
        means.reverse()
        stds.reverse()
        weights.reverse()
    best_threshold = means[0]
    best_gap = float("inf")
    for index in range(10_001):
        threshold = means[0] + (means[1] - means[0]) * index / 10_000
        gap = abs(
            weights[0] * normal_pdf(threshold, means[0], stds[0])
            - weights[1] * normal_pdf(threshold, means[1], stds[1])
        )
        if gap < best_gap:
            best_threshold = threshold
            best_gap = gap
    return best_threshold, weights, means, stds


def report(name: str, pairs: list[tuple[float, int]], threshold: float) -> None:
    precision, recall, f1 = metrics(pairs, threshold)
    print(f"{name:18s} threshold={threshold:.6f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores_csv", type=Path)
    args = parser.parse_args()

    pairs = load_scores(args.scores_csv)
    values = sorted(score for score, _ in pairs)
    report("q60", pairs, quantile(values, 0.60))
    report("otsu", pairs, otsu_threshold(values))
    threshold, weights, means, stds = gmm_intersection(values)
    report("gmm_intersection", pairs, threshold)
    print(f"gmm weights={[round(value, 4) for value in weights]}")
    print(f"gmm means={[round(value, 6) for value in means]}")
    print(f"gmm stds={[round(value, 6) for value in stds]}")


if __name__ == "__main__":
    main()
