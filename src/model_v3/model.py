from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import ModelHubMixin
from safetensors.torch import load_file, save_file
from torch import nn

from .backbone import TSFormer


def _required_int(config: dict[str, Any], key: str) -> int:
    if key not in config:
        raise KeyError(f"Expected model.{key} in model_v3 config")
    return int(config[key])


def _required_heads(config: dict[str, Any]) -> int:
    if "num_heads" in config:
        return int(config["num_heads"])
    if "n_heads" in config:
        return int(config["n_heads"])
    raise KeyError("Expected model.num_heads or model.n_heads in model_v3 config")


def _required_ffn(config: dict[str, Any]) -> int:
    if "d_ff" in config:
        return int(config["d_ff"])
    if "d_ffn" in config:
        return int(config["d_ffn"])
    raise KeyError("Expected model.d_ff or model.d_ffn in model_v3 config")


def _target_fields(config: dict[str, Any]) -> list[str]:
    data_cfg = config["data"]
    fields = data_cfg.get("target_fields", config.get("features"))
    if not fields:
        raise KeyError("Expected non-empty data.target_fields or top-level features in model_v3 config")
    return list(fields)


def build_model_config(config: dict[str, Any]) -> dict[str, Any]:
    if "data" not in config or "model" not in config:
        return dict(config)

    data_cfg = config["data"]
    model_cfg = config["model"]
    seq_len = int(data_cfg["seq_len"])
    patch_size = _required_int(model_cfg, "patch_size")
    if patch_size <= 0:
        raise ValueError(f"model.patch_size must be positive, got {patch_size}")
    if seq_len % patch_size != 0:
        raise ValueError(f"data.seq_len ({seq_len}) must be divisible by model.patch_size ({patch_size})")
    return {
        "model_type": "model_v3",
        "model_id": str(config.get("model_id", "model-v3")),
        "seq_len": seq_len,
        "patch_size": patch_size,
        "input_channels": len(_target_fields(config)),
        "d_model": _required_int(model_cfg, "d_model"),
        "n_heads": _required_heads(model_cfg),
        "num_layers": _required_int(model_cfg, "num_layers"),
        "d_ff": _required_ffn(model_cfg),
        "dropout": float(model_cfg.get("dropout", 0.1)),
        "rope_base": float(model_cfg.get("rope_base", model_cfg.get("rope_theta", 10_000.0))),
        "causal": bool(model_cfg.get("causal", False)),
    }


class Model(nn.Module, ModelHubMixin):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = build_model_config(config)
        model_keys = {
            "patch_size",
            "input_channels",
            "d_model",
            "n_heads",
            "num_layers",
            "d_ff",
            "dropout",
            "rope_base",
            "causal",
        }
        self.model = TSFormer(**{key: self.config[key] for key in model_keys})

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
    ) -> "Model":
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
