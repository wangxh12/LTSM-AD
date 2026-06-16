from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from safetensors.torch import save_file

from src.data import PretrainDataModule
from src.model import PretrainLitModule, ReconstructionTransformer
from src.model_timesbert import TimesBERTModel, TimesBERTPretrainLitModule
from src.scripts.utils import load_config, make_run_dir, save_config, save_json, seed_everything


def _json_dump(payload: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_type: str
    Model: type[torch.nn.Module]
    ModelForPreTraining: type[L.LightningModule]
    default_model_config: dict[str, Any]
    default_pretraining_config: dict[str, Any]


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "timesbert-base": ModelSpec(
        model_id="timesbert-base",
        model_type="timesbert",
        Model=TimesBERTModel,
        ModelForPreTraining=TimesBERTPretrainLitModule,
        default_model_config={
            "patch_len": 4,
            "d_model": 128,
            "n_heads": 4,
            "num_layers": 4,
            "d_ffn": 512,
            "dropout": 0.1,
            "activation": "gelu",
            "norm_first": True,
        },
        default_pretraining_config={
            "task": "mpm",
            "mask_ratio": 0.25,
        },
    ),
    "pointwise-small": ModelSpec(
        model_id="pointwise-small",
        model_type="pointwise",
        Model=ReconstructionTransformer,
        ModelForPreTraining=PretrainLitModule,
        default_model_config={
            "d_model": 128,
            "n_heads": 4,
            "num_layers": 2,
            "d_ffn": 256,
            "dropout": 0.1,
        },
        default_pretraining_config={
            "task": "masked_reconstruction",
            "mask_ratio": 0.25,
            "mask_unit_size": 4,
        },
    ),
}
MODEL_ALIASES = {
    "timesbert": "timesbert-base",
    "v2": "timesbert-base",
    "default": "pointwise-small",
    "pointwise": "pointwise-small",
}


def model_dispatcher(model_id: str | None) -> ModelSpec:
    resolved_id = model_id or "pointwise-small"
    resolved_id = MODEL_ALIASES.get(resolved_id, resolved_id)
    if resolved_id not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model_id={model_id!r}. Supported model ids: {supported}")
    return MODEL_REGISTRY[resolved_id]


def target_fields(config: dict[str, Any]) -> list[str]:
    data_cfg = config.get("data", {})
    fields = data_cfg.get("target_fields", config.get("features"))
    if not fields:
        raise KeyError("Expected non-empty data.target_fields or top-level features in config")
    return list(fields)


def model_config(config: dict[str, Any], spec: ModelSpec | None = None) -> dict[str, Any]:
    data_cfg = config["data"]
    spec = spec or model_dispatcher(config.get("model_id") or config.get("model_version"))
    merged = dict(spec.default_model_config)
    merged.update(config.get("model", {}))
    merged["seq_len"] = int(data_cfg["seq_len"])
    merged["num_features"] = len(target_fields(config))
    return merged


def pretraining_config(config: dict[str, Any], spec: ModelSpec | None = None) -> dict[str, Any]:
    spec = spec or model_dispatcher(config.get("model_id") or config.get("model_version"))
    merged = dict(spec.default_pretraining_config)
    merged.update(config.get("pretraining", {}))
    trainer_cfg = config.get("trainer", {})
    # Backward compatibility with the temporary old config.
    if "mask_ratio" in trainer_cfg and "mask_ratio" not in config.get("pretraining", {}):
        merged["mask_ratio"] = trainer_cfg["mask_ratio"]
    return merged


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
    seed_everything(int(config.get("seed", 42)))
    spec = model_dispatcher(config.get("model_id") or config.get("model_version"))
    mcfg = model_config(config, spec)
    pcfg = pretraining_config(config, spec)
    opt_cfg = config.get("optimization", {})
    loss = str(opt_cfg.get("loss", pcfg.get("loss", "mse")))

    if spec.model_type == "timesbert":
        module = spec.ModelForPreTraining(
            model=mcfg,
            lr=float(opt_cfg.get("lr", 2e-4)),
            weight_decay=float(opt_cfg.get("weight_decay", 1e-2)),
            loss=loss,
            mask_ratio=float(pcfg.get("mask_ratio", 0.25)),
        )
    elif spec.model_type == "pointwise":
        module = spec.ModelForPreTraining(
            model=mcfg,
            lr=float(opt_cfg.get("lr", 1e-3)),
            weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
            loss=loss,
            mask_ratio=float(pcfg.get("mask_ratio", 0.25)),
            mask_unit_size=int(pcfg.get("mask_unit_size", 4)),
        )
    else:
        raise ValueError(f"Unsupported model type for pretraining: {spec.model_type}")

    device = config.get("device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    module.to(device)
    return module


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
    # trainer, checkpoint = build_trainer(config)
    trainer.fit(lightning_module, datamodule=datamodule)
    best_checkpoint = checkpoint.best_model_path or None
    best_score = float(checkpoint.best_model_score.detach().cpu()) if checkpoint.best_model_score is not None else None
    return lightning_module, best_checkpoint, best_score


def load_pretraining_module(config: dict[str, Any], checkpoint_path: str | Path) -> L.LightningModule:
    spec = model_dispatcher(config.get("model_id") or config.get("model_version"))
    return spec.ModelForPreTraining.load_from_checkpoint(checkpoint_path, map_location="cpu")

def export_pretrained_model(
    module: L.LightningModule,
    config: dict[str, Any],
    checkpoint_path: str | None,
    best_val_loss: float | None,
) -> Path:
    export_dir = Path(config["export"]["save_dir"])
    export_dir.mkdir(parents=True, exist_ok=True)

    spec = model_dispatcher(config.get("model_id") or config.get("model_version"))
    backbone = module.model
    config_payload = {
        "model_id": spec.model_id,
        "model_type": spec.model_type,
        **model_config(config, spec),
    }

    if hasattr(backbone, "save_pretrained") and spec.model_type == "timesbert":
        backbone.save_pretrained(export_dir)
        # Ensure model_id is included even if the model itself only saves constructor args.
        _json_dump(config_payload, export_dir / "config.json")
    else:
        _json_dump(config_payload, export_dir / "config.json")
        save_file(backbone.state_dict(), export_dir / "model.safetensors")

    metadata = {
        "model_id": spec.model_id,
        "model_type": spec.model_type,
        "checkpoint_type": "backbone",
        "source_checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "best_val_loss": best_val_loss,
        "pretraining": pretraining_config(config, spec),
        "target_fields": target_fields(config),
        "seq_len": int(config["data"]["seq_len"]),
        "num_features": len(target_fields(config)),
    }
    _json_dump(metadata, export_dir / "metadata.json")
    save_config(config, export_dir / "training_config.yaml")
    return export_dir