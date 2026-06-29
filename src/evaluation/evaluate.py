from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import lightning as L
import numpy as np
import pandas as pd
from src.scripts.utils import model_version, save_json
from src.visualization import plot_reconstruction, plot_scores
from src.evaluation import (
    best_f1_threshold,
    binary_metrics,
    collect_dataset_point_scores,
    compute_auprc,
    compute_auroc,
    reconstruct_dataset_points,
)
from src.evaluation.scoring import ReconstructionMahalanobisScorer
from src.data.csv_windows import FlightDataset


def _make_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def _centered_nanmean(values: np.ndarray, window: int) -> np.ndarray:
    finite = np.isfinite(values)
    weighted_values = np.where(finite, values, 0.0).astype(np.float64)
    weights = finite.astype(np.float64)
    kernel = np.ones(window, dtype=np.float64)
    sums = np.convolve(weighted_values, kernel, mode="same")
    counts = np.convolve(weights, kernel, mode="same")
    output = np.full(values.shape, np.nan, dtype=np.float32)
    valid = counts > 0
    output[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    return output


def _centered_nanmedian(values: np.ndarray, window: int) -> np.ndarray:
    radius = window // 2
    output = np.full(values.shape, np.nan, dtype=np.float32)
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        selected = values[start:end]
        selected = selected[np.isfinite(selected)]
        if len(selected) > 0:
            output[index] = float(np.median(selected))
    return output


def _apply_residual_filter(scores: np.ndarray, evaluation_cfg: dict) -> np.ndarray:
    filter_cfg = evaluation_cfg.get("residual_filter")
    if filter_cfg is None:
        return scores
    if not isinstance(filter_cfg, dict):
        raise TypeError("evaluation.residual_filter must be a mapping")
    if not bool(filter_cfg.get("enabled", True)):
        return scores

    method = str(filter_cfg.get("method", "moving_average"))
    if method in {"none", "identity"}:
        return scores

    window = int(filter_cfg.get("window", 9))
    if window <= 0 or window % 2 == 0:
        raise ValueError("evaluation.residual_filter.window must be a positive odd integer")

    if method in {"moving_average", "mean"}:
        return _centered_nanmean(scores, window)
    if method == "median":
        return _centered_nanmedian(scores, window)
    raise ValueError(f"Unsupported evaluation.residual_filter.method: {method!r}")


def _build_scorer(
    data,
    evaluation_cfg: dict,
    threshold_dataset,
    threshold_reconstruction_scores: np.ndarray,
) -> ReconstructionMahalanobisScorer | None:
    score_cfg = evaluation_cfg.get("score", {})
    method = str(score_cfg.get("method", "reconstruction"))
    if method == "reconstruction":
        return None
    if method != "reconstruction_mahalanobis":
        raise ValueError(f"Unsupported evaluation.score.method: {method!r}")
    if not isinstance(threshold_dataset, FlightDataset):
        raise TypeError("reconstruction_mahalanobis requires a single WindowDataset threshold split")
    train_series = getattr(data, "train_series", None)
    scaler = getattr(data, "scaler", None)
    if train_series is None or scaler is None:
        raise TypeError("reconstruction_mahalanobis requires a DataModule with train_series and scaler")
    return ReconstructionMahalanobisScorer.fit(
        train_values=scaler.transform(train_series.values),
        validation_values=threshold_dataset.series,
        validation_reconstruction_scores=threshold_reconstruction_scores,
        reconstruction_weight=float(score_cfg.get("reconstruction_weight", 0.5)),
        ridge_multiplier=float(score_cfg.get("ridge_multiplier", 1e-3)),
    )


def evaluate_model(
    model: L.LightningModule,
    data,
    config,
    output_dir: str | Path
):
    device = model.device
    model.to(device)
    model.eval()
    
    dcfg = config.get("data", {})
    evaluation_cfg = config["evaluation"]
    threshold_percentile = float(evaluation_cfg.get("threshold_percentile", 95))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, object]] = []
    test_batch_size = int(dcfg.get("batch_size", 256))
    num_workers = int(dcfg.get("num_workers", 0))
    threshold_dataset = getattr(data, "threshold_dataset", None)
    if threshold_dataset is None:
        raise RuntimeError("DataModule must expose threshold_dataset for threshold calibration")
    threshold_reconstruction_scores = collect_dataset_point_scores(
        model,
        threshold_dataset,
        device=device,
        batch_size=test_batch_size,
        num_workers=num_workers,
    )
    scorer = _build_scorer(data, evaluation_cfg, threshold_dataset, threshold_reconstruction_scores)
    threshold_scores = (
        scorer.score(threshold_reconstruction_scores, threshold_dataset.series)
        if scorer is not None
        else threshold_reconstruction_scores
    )
    threshold_scores = _apply_residual_filter(threshold_scores, evaluation_cfg)
    threshold_scores = threshold_scores[np.isfinite(threshold_scores)]
    threshold = float(np.percentile(threshold_scores, threshold_percentile))
    if scorer is not None:
        save_json(scorer.to_dict(), output_dir / "score_calibration.json")

    test_files = getattr(data, "test_files", None)
    if not test_files:
        raise RuntimeError("DataModule must expose a non-empty test_files list for evaluation")

    for test_path in test_files:
        test_path = Path(test_path)
        dataset_name = test_path.stem
        dataset = data.make_test_dataset(test_path)
        series = data.make_test_series(test_path)
        loader = _make_loader(dataset, batch_size=test_batch_size, num_workers=num_workers)
        point_outputs = reconstruct_dataset_points(model, loader, dataset, device=device)

        target = data.scaler.inverse_transform(point_outputs["target"])
        reconstruction = data.scaler.inverse_transform(point_outputs["reconstruction"])
        raw_scores = point_outputs["scores"]
        scores = raw_scores
        if scorer is not None:
            scores = scorer.score(scores, series.values)
        raw_scores = scores
        scores = _apply_residual_filter(scores, evaluation_cfg)
        labels = point_outputs["labels"]

        fixed = binary_metrics(labels, scores, threshold)
        best = best_f1_threshold(labels, scores)
        row = {
            "dataset": dataset_name,
            "model_version": model_version(config),
            **fixed,
            **best,
            "auroc": compute_auroc(labels, scores),
            "auprc": compute_auprc(labels, scores),
            "anomaly_points": int((labels == 1).sum()),
            "normal_points": int((labels == 0).sum()),
        }
        metrics_rows.append(row)

        dataset_dir = output_dir / "test" / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        save_json(row, dataset_dir / "metrics.json")
        pd.DataFrame(
            {
                "time": series.time if series.time is not None else np.arange(len(scores)),
                "score": scores,
                "raw_score": raw_scores,
                "label": labels,
            }
        ).to_csv(
            dataset_dir / "scores.csv",
            index=False,
        )
        plot_reconstruction(
            output_dir=dataset_dir / "reconstruction",
            dataset_name=dataset_name,
            feature_names=config["data"]["target_fields"],
            target=target,
            reconstruction=reconstruction,
            labels=labels,
            time=series.time,
        )
        plot_scores(
            output_path=dataset_dir / f"{dataset_name}_scores.png",
            dataset_name=dataset_name,
            scores=scores,
            threshold=threshold,
            best_threshold=best["best_threshold"],
            labels=labels,
            time=series.time,
        )

    # save overall metrics
    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False)
    return SimpleNamespace(
        metrics=metrics_rows,
        threshold=threshold,
    )
