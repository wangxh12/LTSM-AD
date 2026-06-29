from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import ModelHubMixin
from safetensors.torch import load_file, save_file
from torch import nn

from .backbone import BlindPatchReconstructionModel


def build_model_config(config: dict[str, Any]) -> dict[str, Any]:
    if "data" not in config or "model" not in config:
        return dict(config)
    data_cfg = config["data"]
    model_cfg = config["model"]
    required = ("patch_len", "d_model", "num_heads", "d_ffn")
    missing = [name for name in required if name not in model_cfg]
    if missing:
        raise KeyError(f"model_toto_blind requires model fields: {missing}")
    return {
        "model_type": "model_toto_blind",
        "model_id": str(config.get("model_id", "toto-blind")),
        "seq_len": int(data_cfg["seq_len"]),
        "num_features": len(data_cfg["target_fields"]),
        "patch_len": int(model_cfg["patch_len"]),
        "d_model": int(model_cfg["d_model"]),
        "num_heads": int(model_cfg["num_heads"]),
        "d_ffn": int(model_cfg["d_ffn"]),
        "dropout": float(model_cfg.get("dropout", 0.1)),
    }


class Model(nn.Module, ModelHubMixin):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = build_model_config(config)
        model_keys = {"seq_len", "num_features", "patch_len", "d_model", "num_heads", "d_ffn", "dropout"}
        self.model = BlindPatchReconstructionModel(**{key: self.config[key] for key in model_keys})

    def _save_pretrained(self, save_directory: Path) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        (save_directory / "config.json").write_text(json.dumps(self.config, indent=2), encoding="utf-8")
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
        model_path = Path(model_id)
        if not model_path.is_dir():
            raise ValueError("model_toto_blind supports local pretrained directories only")
        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        model_kwargs.pop("config", None)
        if model_kwargs:
            config.update(model_kwargs)
        model = cls(config)
        model.load_state_dict(load_file(model_path / "model.safetensors", device=map_location), strict=strict)
        model.eval()
        return model


__all__ = ["Model", "build_model_config"]
