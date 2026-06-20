from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import lightning as L
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from src.data import PretrainDataModule
from src.scripts.utils import load_model_package, save_config, save_json, seed_everything



def target_fields(config: dict[str, Any]) -> list[str]:
    data_cfg = config.get("data", {})
    fields = data_cfg.get("target_fields", config.get("features"))
    if not fields:
        raise KeyError("Expected non-empty data.target_fields or top-level features in config")
    return list(fields)



def get_datamodule(config: dict[str, Any], setup: bool = False) -> PretrainDataModule:
    data_cfg = config["data"]
    dm = PretrainDataModule(
        root=data_cfg.get("root_path", data_cfg.get("root")),
        pattern=data_cfg.get("pattern", "ThorFlight*.csv"),
        feature_names=target_fields(config),
        seq_len=int(data_cfg["seq_len"]),
        stride=int(data_cfg.get("stride", 1)),
        split_ratio=float(data_cfg.get("train_val_split", 0.9)),
        batch_size=int(data_cfg.get("batch_size", 256)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        label_col=data_cfg.get("label_col", "label"),
        time_col=data_cfg.get("timestamp_col", data_cfg.get("time_col", "time")),
        scaler_type=str(data_cfg.get("scaler_type", config.get("scaler_type", "standard"))),
    )
    if setup:
        dm.setup("fit")
    return dm


def init_lightning(config: dict[str, Any]) -> L.LightningModule:
    # Seed
    seed_everything(int(config.get("seed", 42)))
    
    Model, ModelForPreTraining, _ = load_model_package(config.get("model_family", "model_timesbert"))
    
    # Backbone (from config)
    backbone = Model(config).model
    
    # LightningModule
    lightning_module = ModelForPreTraining(
        backbone=backbone,
        config=config,
    )
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    lightning_module.to(device)
    return lightning_module


def train(
    lightning_module: L.LightningModule,
    datamodule: PretrainDataModule,
    config: dict[str, Any],
) -> tuple[L.LightningModule, str | None, float | None]:
    
    trainer_cfg = config.get("trainer", {})
    early_cfg = config.get("early_stopping", {})
    checkpoint_cfg = config.get("checkpoint", {})
    logging_cfg = config.get("logging", {})
    checkpoint_dir = Path(checkpoint_cfg.get("dirpath", checkpoint_cfg.get("root_dir", "checkpoints")))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint callback
    monitor = str(checkpoint_cfg.get("monitor", early_cfg.get("monitor", "val_loss")))
    mode = str(checkpoint_cfg.get("mode", early_cfg.get("mode", "min")))
    checkpoint = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=str(checkpoint_cfg.get("filename", "{epoch:03d}-{val_loss:.6f}")),
        monitor=monitor,
        mode=mode,
        save_top_k=int(checkpoint_cfg.get("save_top_k", 1)),
        save_last=bool(checkpoint_cfg.get("save_last", True)),
    )
    callbacks: list[L.Callback] = [
        TQDMProgressBar(refresh_rate=int(trainer_cfg.get("refresh_rate", 1))),
        checkpoint,
    ]
    if bool(early_cfg.get("enabled", True)):
        callbacks.append(
            EarlyStopping(
                monitor=str(early_cfg.get("monitor", monitor)),
                mode=str(early_cfg.get("mode", mode)),
                patience=int(early_cfg.get("patience", trainer_cfg.get("early_stopping_patience", 8))),
                min_delta=float(early_cfg.get("min_delta", 0.0)),
            )
        )

    # TensorBoard logger
    logger = TensorBoardLogger(
        save_dir=str(logging_cfg.get("save_dir", "lightning_logs")),
        name=str(logging_cfg.get("name", "pretrain")),
    )
    
    trainer_kwargs: dict[str, Any] = {
        "accelerator": trainer_cfg.get("accelerator", "auto"),
        "devices": trainer_cfg.get("devices", 1),
        "precision": trainer_cfg.get("precision", "32-true"),
        "callbacks": callbacks,
        "logger": logger,
        "log_every_n_steps": int(trainer_cfg.get("log_every_n_steps", 20)),
        "deterministic": bool(trainer_cfg.get("deterministic", False)),
        "num_sanity_val_steps": int(trainer_cfg.get("num_sanity_val_steps", 0)),
    }
    if "max_steps" in trainer_cfg:
        trainer_kwargs["max_steps"] = int(trainer_cfg["max_steps"])
    else:
        trainer_kwargs["max_epochs"] = int(trainer_cfg.get("max_epochs", 50))
    for key in ("val_check_interval", "check_val_every_n_epoch", "limit_train_batches", "limit_val_batches"):
        if trainer_cfg.get(key) is not None:
            trainer_kwargs[key] = trainer_cfg[key]

    trainer = Trainer(**trainer_kwargs)
    
    trainer.fit(lightning_module, datamodule=datamodule)
    best_checkpoint = checkpoint.best_model_path or None
    best_score = float(checkpoint.best_model_score.detach().cpu()) if checkpoint.best_model_score is not None else None
    return lightning_module, best_checkpoint, best_score


def save_pretrained(
    module: L.LightningModule,
    config: dict[str, Any],
    checkpoint_path: str | Path | None,
    best_val_loss: float | None,
    dataset_name: str | None = None,
    checkpoint_dir: str | Path | None = None,
    logging_dir: str | Path | None = None,
    csv_paths: Iterable[str | Path] | None = None,
) -> Path:
    model_id = config.get("model_id")
    assert model_id, "model_id must be specified in config for saving pretrained model"
    
    export_dir = Path(config["export"]["save_dir"])
    pretrained_model_dir = export_dir / model_id
    print(f"Exporting pretrained model to {pretrained_model_dir}")
    # exit(0)
    pretrained_model_dir.mkdir(parents=True, exist_ok=True)

    Model, _, _ = load_model_package(config.get("model_family", "model_timesbert"))

    # 1. 重新构造 wrapper
    wrapper = Model(config)

    # 2. 把 Lightning 中训练好的 backbone 权重塞回 wrapper.model
    wrapper.model.load_state_dict(module.model.state_dict(), strict=True)

    # 3. 保存成 Hugging Face / Toto 风格
    wrapper.save_pretrained(pretrained_model_dir)

    # 4. 保存额外训练信息
    checkpoint_cfg = config.get("checkpoint", {})
    logging_cfg = config.get("logging", {})
    if checkpoint_dir is None and checkpoint_cfg.get("dirpath") is not None:
        checkpoint_dir = checkpoint_cfg["dirpath"]
    if logging_dir is None and logging_cfg:
        logging_dir = Path(logging_cfg.get("save_dir", "lightning_logs")) / str(logging_cfg.get("name", "pretrain"))

    metadata = {
        "model_id": model_id,
        "dataset_name": dataset_name,
        "best_checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "best_val_loss": best_val_loss,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "logging_dir": str(logging_dir) if logging_dir is not None else None,
        "csv_paths": [str(path) for path in csv_paths] if csv_paths is not None else [],
        "target_fields": target_fields(config),
        "seq_len": int(config["data"]["seq_len"]),
        "num_features": len(target_fields(config)),
    }
    save_json(metadata, pretrained_model_dir / "metadata.json")
    save_config(config, pretrained_model_dir / "training_config.yaml")

    # 5. 如果有归一化统计量，额外保存
    # if "normalization" in config:
    #     save_json(config["normalization"], pretrained_model_dir / "stats.json")

    return pretrained_model_dir

def load_pretraining_module(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> L.LightningModule:
    """Load a pretrained LightningModule from a Lightning checkpoint."""
    Model, ModelForPreTraining, _ = load_model_package(config.get("model_family", "model_timesbert"))
    wrapper = Model(config)

    module = ModelForPreTraining.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        backbone=wrapper.model,
        config=config,
        map_location=map_location,
    )
    module.eval()
    return module
