from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from src.data import PretrainDataModule
from src.model_timesbert.lightning import ModelForPreTraining
from src.model_timesbert.timesbert import Model
from src.scripts.utils import load_config, make_run_dir, save_config, save_json, seed_everything


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
    
    run_dir = Path(config["runtime"]["run_dir"])
    trainer_cfg = config.get("trainer", {})
    early_cfg = config.get("early_stopping", {})
    checkpoint_cfg = config.get("checkpoint", {})
    logging_cfg = config.get("logging", {})

    # Checkpoint callback
    monitor = str(checkpoint_cfg.get("monitor", early_cfg.get("monitor", "val_loss")))
    mode = str(checkpoint_cfg.get("mode", early_cfg.get("mode", "min")))
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
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
        save_dir=str(logging_cfg.get("save_dir", run_dir / "logs")),
        name=str(logging_cfg.get("name", "pretrain")),
    )
    
    trainer = Trainer(
        max_epochs=int(trainer_cfg.get("max_epochs", 50)),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", 1),
        precision=trainer_cfg.get("precision", "32-true"),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 20)),
        deterministic=bool(trainer_cfg.get("deterministic", False)),
        num_sanity_val_steps=int(trainer_cfg.get("num_sanity_val_steps", 0)),
        enable_progress_bar=bool(trainer_cfg.get("enable_progress_bar", True)),
    )
    
    trainer.fit(lightning_module, datamodule=datamodule)
    best_checkpoint = checkpoint.best_model_path or None
    best_score = float(checkpoint.best_model_score.detach().cpu()) if checkpoint.best_model_score is not None else None
    return lightning_module, best_checkpoint, best_score


def save_pretrained(
    module: ModelForPreTraining,
    config: dict[str, Any],
    checkpoint_path: str | Path | None,
    best_val_loss: float | None,
) -> Path:
    model_id = config.get("model_id")
    assert model_id, "model_id must be specified in config for saving pretrained model"
    
    export_dir = Path(config["export"]["save_dir"])
    pretrained_model_dir = export_dir / model_id
    print(f"Exporting pretrained model to {pretrained_model_dir}")
    # exit(0)
    pretrained_model_dir.mkdir(parents=True, exist_ok=True)

    # 1. 重新构造 wrapper
    wrapper = Model(config)

    # 2. 把 Lightning 中训练好的 backbone 权重塞回 wrapper.model
    wrapper.model.load_state_dict(module.model.state_dict(), strict=True)

    # 3. 保存成 Hugging Face / Toto 风格
    wrapper.save_pretrained(pretrained_model_dir)

    # 4. 保存额外训练信息
    metadata = {
        "model_id": model_id,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "best_val_loss": best_val_loss,
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
) -> ModelForPreTraining:
    """Load a pretrained LightningModule from a Lightning checkpoint."""
    wrapper = Model(config)

    module = ModelForPreTraining.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        backbone=wrapper.model,
        config=config,
        map_location=map_location,
    )
    module.eval()
    return module