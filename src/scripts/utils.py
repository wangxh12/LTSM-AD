from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

try:
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
except ImportError:  # pragma: no cover - kept for older environments
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_config(config: dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(root: str | Path, prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def model_config(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config["data"]
    model_cfg = dict(config["model"])
    model_cfg["seq_len"] = int(data_cfg["seq_len"])
    feature_names = data_cfg.get("target_fields", config.get("features"))
    if feature_names is None:
        raise KeyError("Expected data.target_fields or top-level features in config")
    model_cfg["num_features"] = len(feature_names)
    return model_cfg


def model_version(config: dict[str, Any]) -> str:
    return str(config.get("model_version", "default")).lower()


def load_lightning_weights(module: torch.nn.Module, checkpoint_path: str | Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys when loading {checkpoint_path}: {missing}")
    if unexpected:
        print(f"Unexpected keys when loading {checkpoint_path}: {unexpected}")


def device_from_trainer_config(config: dict[str, Any]) -> torch.device:
    accelerator = str(config.get("trainer", {}).get("accelerator", "auto"))
    if accelerator in {"cuda", "gpu", "auto"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
