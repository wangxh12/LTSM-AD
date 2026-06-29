import argparse
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml
import lightning as L
from datasets import load_dataset
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from src.data.csv_windows import FinetuneDataModule
from src.scripts.utils import save_json

from src.model_v1 import ReconstructionLightningModule as V1ReconstructionLitModule
from src.model_timesbert import TimesBERTLitModule
from src.model import ReconstructionLitModule
from src.scripts.utils import load_lightning_weights, model_config, model_version

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}

def _build_module(config: Dict[str, Any]) -> L.LightningModule:
    version = model_version(config)
    opt_cfg = config.get("optimization", {})

    lr = float(opt_cfg.get("lr", 5e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-4))
    loss = str(opt_cfg.get("loss", "mse"))

    if version == "v1":
        return V1ReconstructionLitModule(
            feature_columns=config["data"]["target_fields"],
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

def init_lightning(config: Dict[str, Any]) -> L.lightningModule:
    # Seed
    seed = int(config.get("seed", 42))
    seed_everything(seed, workers=True)

    # Backbone
    model_id = config.get("pretrained_model", "Datadog/Toto-Open-Base-1.0")
    pretrained_backbone = Toto.from_pretrained(model_id).model

    model = _build_module(config)
    
    # LightningModule params
    mcfg = config.get("model", {})
    dcfg = config.get("data", {})

    lightning_module = TotoForFinetuning(
        pretrained_backbone=pretrained_backbone,
        val_prediction_len=int(mcfg.get("val_prediction_len", 96)),
        stable_steps=int(mcfg.get("stable_steps", 1000)),
        decay_steps=int(mcfg.get("decay_steps", 1000)),
        warmup_steps=int(mcfg.get("warmup_steps", 200)),
        lr=float(mcfg.get("lr", 1e-4)),
        min_lr=float(mcfg.get("min_lr", 1e-5)),
        add_exogenous_features=bool(dcfg.get("add_exogenous_features", False)),
    )

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    lightning_module.to(device)
    return lightning_module


def get_datamodule(config: Dict[str, Any], setup: bool = False) -> FinetuneDataModule:
    data_cfg = config["data"]
    dm = FinetuneDataModule(
        train_path=data_cfg["train_path"],
        test_paths=data_cfg["test_paths"],
        feature_names=data_cfg["target_fields"],
        seq_len=int(data_cfg["seq_len"]),
        stride=int(data_cfg.get("stride", 1)),
        eval_stride=int(data_cfg.get("eval_stride", data_cfg.get("stride", 1))),
        split_ratio=float(data_cfg.get("train_val_split", 0.9)),
        batch_size=int(data_cfg.get("batch_size", data_cfg.get("batch_size", 256))),
        num_workers=int(data_cfg.get("num_workers", 0)),
        label_col=data_cfg.get("label_col", "label"),
        time_col=data_cfg.get("timestamp_col", "time"),
        scaler_type=str(data_cfg.get("scaler_type", config.get("scaler_type", "standard"))),
    )
    if setup:
        dm.setup(None)
        
    if dm.scaler is None:
        raise RuntimeError("Finetune scaler was not initialized")
    save_json(dm.scaler.to_dict(), config["outputs"]["root_dir"] / "scaler.json")
    return dm


def train(
    lightning_module, datamodule: FinetuneDataModule, config: Dict[str, Any]
):
    tcfg = config.get("trainer", {})
    lcfg = config.get("logging", {})

    # -------------------
    # Progress bar callback
    # -------------------
    callbacks: list[Callback] = [TQDMProgressBar(refresh_rate=int(tcfg.get("refresh_rate", 1)))]
    
    # -------------------
    # Checkpoint callback (optional, controlled via config)
    # -------------------
    cckpt = config.get("checkpoint", {})

    # Only add ModelCheckpoint if any of the relevant options are present
    use_checkpoint = "dirpath" in cckpt.keys()
    checkpoint_callback: ModelCheckpoint | None = None

    if use_checkpoint:
        # check if checkpoint directory already exists if yes then add a suffix to the directory name
        if os.path.exists(cckpt.get("dirpath", "checkpoints")):
            cckpt["dirpath"] = str(cckpt["dirpath"]) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        monitor = cckpt.get("monitor")
        mode = cckpt.get("mode")
        save_top_k_cfg = cckpt.get("save_top_k")

        dirpath = str(cckpt.get("dirpath", "checkpoints"))
        filename = str(cckpt.get("filename", "{epoch}-{step}-{val_loss:.4f}"))

        # Decide checkpointing schedule
        every_n_train_steps = cckpt.get("every_n_train_steps", None)
        every_n_epochs = cckpt.get("every_n_epochs", None)
        train_time_interval_minutes = cckpt.get("train_time_interval_minutes", None)

        if every_n_train_steps is None and every_n_epochs is None and train_time_interval_minutes is None:
            # Set the checkpoint saving after each validation check
            save_on_train_epoch_end = False

        # Convert to proper types or None
        every_n_train_steps = int(every_n_train_steps) if every_n_train_steps is not None else None
        every_n_epochs = int(every_n_epochs) if every_n_epochs is not None else None
        train_time_interval = (
            timedelta(minutes=float(train_time_interval_minutes)) if train_time_interval_minutes is not None else None
        )

        # --- Default behavior when nothing is specified: save ALL checkpoints ---
        if monitor is None and mode is None and save_top_k_cfg is None:
            # "Just save everything"
            monitor_arg = None  # no ranking metric
            mode_arg = "min"  # ignored when monitor=None
            save_top_k_arg = -1  # -1 = save all checkpoints
        else:
            # User configured something -> respect it, with sensible defaults
            monitor_arg = monitor or "val_loss"
            mode_arg = mode or "min"
            save_top_k_arg = int(save_top_k_cfg) if save_top_k_cfg is not None else 1

        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            filename=filename,  # e.g. "{epoch}-{step}-{val_loss:.4f}"
            monitor=monitor_arg,
            mode=mode_arg,
            save_top_k=save_top_k_arg,
            every_n_train_steps=every_n_train_steps,
            every_n_epochs=every_n_epochs,
            train_time_interval=train_time_interval,
            save_on_train_epoch_end=save_on_train_epoch_end,
        )
        callbacks.append(checkpoint_callback)

    # -------------------
    # TensorBoard logger
    # -------------------
    tb_logger = TensorBoardLogger(
        save_dir=str(lcfg.get("save_dir", "lightning_logs")),
        name=str(lcfg.get("name", "finetuning")),
    )

    # -------------------
    # Trainer kwargs (including validation scheduling)
    # -------------------
    trainer_kwargs: Dict[str, Any] = dict(
        max_steps=int(tcfg.get("max_steps", 3000)),
        log_every_n_steps=int(tcfg.get("log_every_n_steps", 1)),
        num_sanity_val_steps=int(tcfg.get("num_sanity_val_steps", 0)),
        enable_progress_bar=bool(tcfg.get("enable_progress_bar", True)),
        val_check_interval=tcfg.get("val_check_interval", None),
        check_val_every_n_epoch=tcfg.get("check_val_every_n_epoch", None),
        callbacks=callbacks,
        logger=tb_logger,
    )

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(lightning_module, datamodule=datamodule)

    # Extract best checkpoint info from the ModelCheckpoint callback
    best_ckpt_path: str | None = None
    best_score: float | None = None
    if checkpoint_callback is not None:
        best_ckpt_path = checkpoint_callback.best_model_path or None
        best_score = (
            float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score is not None else None
        )

    return lightning_module, best_ckpt_path, best_score


def load_finetuned_module(
    model_id: str,
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
) -> L.LightningModule:
    """
    Load a finetuned model from a checkpoint file.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier for the pretrained model
        (e.g., "Datadog/Toto-Open-Base-1.0").
    checkpoint_path : str
        Path to the Lightning checkpoint file (.ckpt).
    map_location : str | torch.device, default="cpu"
        Device to map the checkpoint tensors to.

    Returns
    -------
    L.LightningModule
        The loaded and eval-ready finetuned model.
    """
    
    # Dispatcher
    Model, ModelForFinetuning = ModelDispatcher(model_id)
    
    # Load base model backbone from HuggingFace
    pretrained_backbone = Model.from_pretrained(model_id).model

    # Load Lightning module from checkpoint
    model = ModelForFinetuning.load_from_checkpoint(  # type: ignore[operator]
        checkpoint_path=checkpoint_path,
        pretrained_backbone=pretrained_backbone,
        map_location=map_location,
    )
    model.eval()
    return model
