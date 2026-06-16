from __future__ import annotations

import argparse
import json
from pathlib import Path
import lightning as L

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.evaluation.evaluate import evaluate_model
from src.scripts import finetune

from src.data import FinetuneDataModule, StandardScaler

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


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune and evaluate UAV reconstruction anomaly detector.")
    parser.add_argument(
        "--config",
        default="src/configs/finetune_config.yaml",
        help="Path to finetuning/evaluation YAML.",
    )
    parser.add_argument("--is_finetuning", type=_bool_arg, default=True)
    # parser.add_argument("--checkpoint", default=None, help="Checkpoint for direct test or override trained checkpoint.")
    parser.add_argument("--pretrained_checkpoint", default=None, help="Optional pretraining checkpoint for finetuning init.")
    args = parser.parse_args()

    # load config
    config = load_config(args.config)
    run_dir = make_run_dir(config["outputs"]["root_dir"], "finetune")
    save_config(config, run_dir / "config.yaml")

    # runtime info
    config.setdefault("runtime", {})
    config["runtime"] = {
        "run_dir": str(run_dir),
    }
    config["runtime"]["checkpoint_path"] = args.pretrained_checkpoint or config.get("pretrained_checkpoint")
    
    # seed
    seed_everything(int(config.get("seed", 42)))
    
    # Initialize Lightning module and datamodule
    lightning_module = finetune.init_lightning(config)
    datamodule = finetune.get_datamodule(config, setup=True)
    
    # Train or run zero-shot
    if args.is_finetuning:
        _, best_ckpt_path, best_val_loss = finetune.train(lightning_module, datamodule, config)

        if best_ckpt_path is None:
            raise RuntimeError("No checkpoint was saved during training. Check checkpoint config.")

        pretrained_model = cast(str, config["pretrained_model"])

        # Load best finetuned model checkpoint
        trained_model = finetune.load_finetuned_module(
            pretrained_model,
            best_ckpt_path,
            lightning_module.device,
        )
        # checkpoint_path = best_ckpt_path
        config["runtime"]["checkpoint_path"] = best_ckpt_path
    else:
        trained_model = lightning_module
        best_val_loss = None

    print("Best validation loss: ", best_val_loss)
    
    # Evaluate model
    result = evaluate_model(trained_model, datamodule, config)

    threshold_percentile = float(config["evaluation"].get("threshold_percentile", 95))
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(config["runtime"].get("checkpoint_path")),
        # "scaler": str(run_dir / "scaler.json"),
        "threshold_percentile": threshold_percentile,
        "threshold": result.threshold,
        "metrics": result.metrics,
    }
    save_json(summary, run_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
