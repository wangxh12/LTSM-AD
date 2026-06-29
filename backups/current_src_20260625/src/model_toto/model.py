from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import ModelHubMixin
from safetensors.torch import load_file, save_file
from torch import nn

from .backbone import TotoReconstructionModel


def _required_int(config: dict[str, Any], key: str) -> int:
    if key not in config:
        raise KeyError(f"Expected model.{key} in model_toto config")
    return int(config[key])


def build_model_config(config: dict[str, Any]) -> dict[str, Any]:
    if "data" in config and "model" in config:
        data_cfg = config["data"]
        model_cfg = config["model"]
        return {
            "model_type": "model_toto",
            "model_id": config.get("model_id", "toto-base"),
            "seq_len": int(data_cfg["seq_len"]),
            "num_features": len(data_cfg["target_fields"]),
            "patch_len": int(model_cfg.get("patch_len", 4)),
            "d_model": _required_int(model_cfg, "d_model"),
            "num_layers": _required_int(model_cfg, "num_layers"),
            "num_heads": _required_int(model_cfg, "num_heads"),
            "d_ffn": _required_int(model_cfg, "d_ffn"),
            "dropout": float(model_cfg.get("dropout", 0.1)),
            "activation": str(model_cfg.get("activation", "gelu")),
        }
    return dict(config)


class Model(nn.Module, ModelHubMixin):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = build_model_config(config)
        model_keys = {
            "seq_len",
            "num_features",
            "patch_len",
            "d_model",
            "num_layers",
            "num_heads",
            "d_ffn",
            "dropout",
            "activation",
        }
        model_kwargs = {key: self.config[key] for key in model_keys}
        self.model = TotoReconstructionModel(**model_kwargs)

    def _save_pretrained(self, save_directory: Path) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        with open(save_directory / "config.json", "w", encoding="utf-8") as handle:
            json.dump(self.config, handle, ensure_ascii=False, indent=2)
        save_file(self.state_dict(), save_directory / "model.safetensors")

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        proxies=None,
        resume_download=None,
        local_files_only: bool = False,
        token: str | bool | None = None,
        map_location: str = "cpu",
        strict: bool = True,
        **model_kwargs,
    ):
        model_id_path = Path(model_id)
        if not model_id_path.is_dir():
            raise ValueError(
                f"Unsupported model_id format: {model_id}. Only local directory paths are supported."
            )

        config_path = model_id_path / "config.json"
        weight_path = model_id_path / "model.safetensors"
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)

        model_kwargs.pop("config", None)
        if model_kwargs:
            config.update(model_kwargs)

        model = cls(config)
        state_dict = load_file(weight_path, device=map_location)
        model.load_state_dict(state_dict, strict=strict)
        model.eval()
        return model


__all__ = ["Model", "build_model_config"]
