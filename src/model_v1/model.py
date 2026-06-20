from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import ModelHubMixin
from safetensors.torch import load_file, save_file
from torch import nn

from .backbone import ReconstructionModel


def build_model_config(config: dict) -> dict:
    if "data" in config and "model" in config:
        data_cfg = config["data"]
        model_cfg = config["model"]
        return {
            "model_type": "model_v1",
            "model_id": config.get("model_id", "model_v1-base"),
            "seq_len": int(data_cfg["seq_len"]),
            "feature_count": len(data_cfg["target_fields"]),
            "hidden_size": int(model_cfg["hidden_size"]),
            "num_layers": int(model_cfg["num_layers"]),
            "num_heads": int(model_cfg["num_heads"]),
            "ffn_size": int(model_cfg["ffn_size"]),
            "dropout": float(model_cfg.get("dropout", 0.1)),
            "rms_norm_eps": float(model_cfg.get("rms_norm_eps", 1e-6)),
            "rope_theta": float(model_cfg.get("rope_theta", 10_000.0)),
        }
    return dict(config)


class Model(nn.Module, ModelHubMixin):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = build_model_config(config)
        model_keys = {
            "feature_count",
            "hidden_size",
            "num_layers",
            "num_heads",
            "ffn_size",
            "dropout",
            "rms_norm_eps",
            "rope_theta",
        }
        model_kwargs = {key: self.config[key] for key in model_keys}
        self.model = ReconstructionModel(**model_kwargs)

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
