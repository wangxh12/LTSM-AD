from __future__ import annotations

import argparse
import json
from pathlib import Path
import lightning as L

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import FinetuneDataModule, StandardScaler
from src.evaluation import (
    best_f1_threshold,
    binary_metrics,
    collect_window_point_scores,
    compute_auprc,
    compute_auroc,
    reconstruct_dataset_points,
)
from src.model_v1 import ReconstructionLightningModule as V1ReconstructionLitModule
from src.model_timesbert import TimesBERTLitModule
from src.model import ReconstructionLitModule
from src.scripts.utils import (
    device_from_trainer_config,
    load_config,
    load_lightning_weights,
    make_run_dir,
    model_config,
    model_version,
    save_config,
    save_json,
    seed_everything,
    trainer_and_checkpoint,
)
from src.visualization import plot_reconstruction, plot_scores


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value}")


def _make_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def _build_module(config: dict, lr: float, weight_decay: float, loss: str):
    version = model_version(config)
    if version == "v1":
        return V1ReconstructionLitModule(
            feature_columns=config["features"],
            model_config=dict(config["model"]),
            objective="finetune",
            learning_rate=lr,
            weight_decay=weight_decay,
        )
    if version in {"v2", "timesbert"}:
        return TimesBERTLitModule(
            model=model_config(config),
            lr=lr,
            weight_decay=weight_decay,
            loss=loss,
        )
    if version in {"default", "pointwise"}:
        return ReconstructionLitModule(
            model=model_config(config),
            lr=lr,
            weight_decay=weight_decay,
            loss=loss,
        )
    raise ValueError(f"Unsupported model_version: {version}")


def _module_lr(config: dict) -> float:
    return float(config["finetune"].get("lr", config["optimization"].get("lr", 5e-4)))


def _module_weight_decay(config: dict) -> float:
    return float(config["optimization"].get("weight_decay", 1e-4))


def _module_loss(config: dict) -> str:
    return str(config["optimization"].get("loss", "mse"))


def _load_module(config: dict, checkpoint_path: str | Path):
    version = model_version(config)
    if version == "v1":
        return V1ReconstructionLitModule.load_from_checkpoint(checkpoint_path)
    if version in {"v2", "timesbert"}:
        return TimesBERTLitModule.load_from_checkpoint(checkpoint_path)
    if version in {"default", "pointwise"}:
        return ReconstructionLitModule.load_from_checkpoint(checkpoint_path)
    raise ValueError(f"Unsupported model_version: {version}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune and evaluate UAV reconstruction anomaly detector.")
    parser.add_argument(
        "--config",
        default="src/configs/finetune_config.yaml",
        help="Path to finetuning/evaluation YAML.",
    )
    parser.add_argument("--is_finetuning", type=_bool_arg, default=True)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint for direct test or override trained checkpoint.")
    parser.add_argument("--pretrained_checkpoint", default=None, help="Optional pretraining checkpoint for finetuning init.")
    parser.add_argument("--scaler", default=None, help="Optional scaler JSON for direct test.")
    args = parser.parse_args()

    # load config
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    run_dir = make_run_dir(config["outputs"]["root_dir"], "finetune")
    save_config(config, run_dir / "config.yaml")

    # data module setup
    data_cfg = config["data"]
    dm = FinetuneDataModule(
        train_path=data_cfg["train_path"],
        test_paths=data_cfg["test_paths"],
        feature_names=config["features"],
        seq_len=int(data_cfg["seq_len"]),
        stride=int(data_cfg.get("stride", 1)),
        eval_stride=int(data_cfg.get("eval_stride", data_cfg.get("stride", 1))),
        split_ratio=float(data_cfg.get("train_val_split", 0.9)),
        batch_size=int(data_cfg.get("batch_size", data_cfg.get("batch_size", 256))),
        num_workers=int(data_cfg.get("num_workers", 0)),
        label_col=data_cfg.get("label_col", "label"),
        time_col=data_cfg.get("time_col", "time"),
        scaler_type=str(data_cfg.get("scaler_type", config.get("scaler_type", "standard"))),
    )
    dm.setup("fit")
    if dm.scaler is None:
        raise RuntimeError("Finetune scaler was not initialized")
    save_json(dm.scaler.to_dict(), run_dir / "scaler.json")

    # model and trainer setup
    checkpoint_path = args.checkpoint
    module = None
    if args.is_finetuning:
        # finetune
        module: L.LightningModule = _build_module(
            config=config,
            lr=_module_lr(config),
            weight_decay=_module_weight_decay(config),
            loss=_module_loss(config),
        )
        pretrained_checkpoint = args.pretrained_checkpoint or config.get("pretrained_checkpoint")
        if pretrained_checkpoint:
            load_lightning_weights(module, pretrained_checkpoint)
        trainer_cfg = dict(config["trainer"])
        trainer_cfg.update(data_cfg.get("trainer", {}))
        trainer, checkpoint = trainer_and_checkpoint(run_dir, trainer_cfg)
        
        # fit
        trainer.fit(module, datamodule=dm)
        checkpoint_path = checkpoint_path or checkpoint.best_model_path
        if not checkpoint_path:
            checkpoint_path = str(run_dir / "checkpoints" / "last.ckpt")

    if not checkpoint_path:
        checkpoint_path = data_cfg.get("checkpoint")
    if not checkpoint_path and not args.is_finetuning:
        pretrained_checkpoint = args.pretrained_checkpoint or config.get("pretrained_checkpoint")
        if pretrained_checkpoint:
            module = _build_module(
                config=config,
                lr=_module_lr(config),
                weight_decay=_module_weight_decay(config),
                loss=_module_loss(config),
            )
            load_lightning_weights(module, pretrained_checkpoint)
            checkpoint_path = pretrained_checkpoint
    if not checkpoint_path:
        raise ValueError("A checkpoint is required when --is_finetuning false")

    if module is None:
        module = _load_module(config, checkpoint_path)
    device = device_from_trainer_config(config)
    module.to(device)
    module.eval()

    threshold_loader = dm.threshold_dataloader()
    train_scores = collect_window_point_scores(module, threshold_loader, device=device)
    train_scores = train_scores[np.isfinite(train_scores)]
    threshold_percentile = float(config["evaluation"].get("threshold_percentile", 95))
    threshold = float(np.percentile(train_scores, threshold_percentile))

    metrics_rows: list[dict[str, object]] = []
    test_batch_size = int(data_cfg.get("batch_size", 256))
    num_workers = int(data_cfg.get("num_workers", 0))
    for test_path in data_cfg["test_paths"]:
        test_path = Path(test_path)
        dataset_name = test_path.stem
        dataset = dm.make_test_dataset(test_path)
        series = dm.make_test_series(test_path)
        loader = _make_loader(dataset, batch_size=test_batch_size, num_workers=num_workers)
        point_outputs = reconstruct_dataset_points(module, loader, dataset, device=device)

        target = dm.scaler.inverse_transform(point_outputs["target"])
        reconstruction = dm.scaler.inverse_transform(point_outputs["reconstruction"])
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
            "checkpoint": str(checkpoint_path),
        }
        metrics_rows.append(row)

        dataset_dir = run_dir / "test" / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        save_json(row, dataset_dir / "metrics.json")
        pd.DataFrame({"time": series.time if series.time is not None else np.arange(len(scores)), "score": scores, "label": labels}).to_csv(
            dataset_dir / "scores.csv",
            index=False,
        )
        plot_reconstruction(
            output_dir=dataset_dir / "reconstruction",
            dataset_name=dataset_name,
            feature_names=config["features"],
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

    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False)
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "scaler": str(run_dir / "scaler.json"),
        "threshold_percentile": threshold_percentile,
        "threshold": threshold,
        "metrics": metrics_rows,
    }
    save_json(summary, run_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
