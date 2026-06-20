import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml
import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from src.data.finetune_datamodule import FinetuneDataModule
from src.scripts.utils import load_model_package, save_json

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def init_lightning(config: Dict[str, Any]) -> L.LightningModule:
    # Seed
    seed = int(config.get("seed", 42))
    seed_everything(seed, workers=True)
    
    Model, _, ModelForFinetuning = load_model_package(config.get("model_family", "model_timesbert"))

    # Backbone
    model_id = config.get("pretrained_model", "PretrainedModel/timebert-base")
    pretrained_backbone = Model.from_pretrained(model_id).model
    
    # LightningModule
    lightning_module = ModelForFinetuning(
        pretrained_backbone=pretrained_backbone,
        config=config,
    )

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    lightning_module.to(device)
    return lightning_module


def get_datamodule(
    config: Dict[str, Any],
    setup: bool = False,
    output_dir: str | Path | None = None,
) -> FinetuneDataModule:
    dm = FinetuneDataModule.from_config(config)
    if setup:
        dm.setup(None)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(dm.scalers_to_dict(), output_dir / "scalers.json")
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
        save_on_train_epoch_end = None

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
            save_last=bool(cckpt.get("save_last", False)),
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
        log_every_n_steps=int(tcfg.get("log_every_n_steps", 1)),
        num_sanity_val_steps=int(tcfg.get("num_sanity_val_steps", 0)),
        val_check_interval=tcfg.get("val_check_interval", None),
        check_val_every_n_epoch=tcfg.get("check_val_every_n_epoch", None),
        callbacks=callbacks,
        logger=tb_logger,
    )
    if "max_steps" in tcfg:
        trainer_kwargs["max_steps"] = int(tcfg["max_steps"])
    else:
        trainer_kwargs["max_epochs"] = int(tcfg.get("max_epochs", 1))
    if "accelerator" in tcfg:
        trainer_kwargs["accelerator"] = tcfg["accelerator"]
    if "devices" in tcfg:
        trainer_kwargs["devices"] = tcfg["devices"]
    if "precision" in tcfg:
        trainer_kwargs["precision"] = tcfg["precision"]
    if "deterministic" in tcfg:
        trainer_kwargs["deterministic"] = bool(tcfg["deterministic"])

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
    pretrained_model: str,
    checkpoint_path: str,
    config: Dict[str, Any],
    map_location: str | torch.device = "cpu",
) -> L.LightningModule:
    """
    Load a finetuned model from a checkpoint file.

    Parameters
    ----------
    pretrained_model : str
        Local pretrained model directory.
    checkpoint_path : str
        Path to the Lightning checkpoint file (.ckpt).
    map_location : str | torch.device, default="cpu"
        Device to map the checkpoint tensors to.

    Returns
    -------
    L.LightningModule
        The loaded and eval-ready finetuned model.
    """
    
    Model, _, ModelForFinetuning = load_model_package(config.get("model_family", "model_timesbert"))
    pretrained_backbone = Model.from_pretrained(pretrained_model).model

    # Load Lightning module from checkpoint
    model = ModelForFinetuning.load_from_checkpoint(  # type: ignore[operator]
        checkpoint_path=checkpoint_path,
        pretrained_backbone=pretrained_backbone,
        config=config,
        map_location=map_location,
    )
    model.eval()
    return model
