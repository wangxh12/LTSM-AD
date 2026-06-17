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
    collect_window_point_scores,
    compute_auprc,
    compute_auroc,
    reconstruct_dataset_points,
)

def _make_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
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
    
    threshold_loader = data.threshold_dataloader()
    train_scores = collect_window_point_scores(model, threshold_loader, device=device)
    train_scores = train_scores[np.isfinite(train_scores)]
    threshold_percentile = float(config["evaluation"].get("threshold_percentile", 95))
    threshold = float(np.percentile(train_scores, threshold_percentile))
    
    dcfg = config.get("data", {})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, object]] = []
    test_batch_size = int(dcfg.get("batch_size", 256))
    num_workers = int(dcfg.get("num_workers", 0))
    for test_path in dcfg["test_paths"]:
        test_path = Path(test_path)
        dataset_name = test_path.stem
        dataset = data.make_test_dataset(test_path)
        series = data.make_test_series(test_path)
        loader = _make_loader(dataset, batch_size=test_batch_size, num_workers=num_workers)
        point_outputs = reconstruct_dataset_points(model, loader, dataset, device=device)

        target = data.scaler.inverse_transform(point_outputs["target"])
        reconstruction = data.scaler.inverse_transform(point_outputs["reconstruction"])
        scores = point_outputs["scores"]
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
        pd.DataFrame({"time": series.time if series.time is not None else np.arange(len(scores)), "score": scores, "label": labels}).to_csv(
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
