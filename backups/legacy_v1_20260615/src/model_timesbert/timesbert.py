from __future__ import annotations

import json
import math
from pathlib import Path
from safetensors.torch import save_file, load_file
from huggingface_hub import ModelHubMixin
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from src.model_timesbert.backbone import TimesBERT

def build_model_config(config: dict) -> dict:
    if "data" in config and "model" in config:
        data_cfg = config["data"]
        model_cfg = config["model"]
        return {
            "model_type": "timesbert",
            "model_id": config.get("model_id", "timebert-base"),
            "seq_len": int(data_cfg["seq_len"]),
            "num_features": len(data_cfg["target_fields"]),
            "patch_len": int(model_cfg.get("patch_len", 4)),
            "d_model": int(model_cfg["d_model"]),
            "num_layers": int(model_cfg["num_layers"]),
            "n_heads": int(model_cfg["n_heads"]),
            "d_ffn": int(model_cfg.get("d_ffn")),
            "dropout": float(model_cfg.get("dropout", 0.1)),
            "activation": str(model_cfg.get("activation", "gelu")),
            "norm_first": bool(model_cfg.get("norm_first", False)),
        }

    return dict(config)

class Model(nn.Module, ModelHubMixin):
    """
    PyTorch module for timesbert .

    Parameters
    ----------
    **model_kwargs
        Additional keyword arguments to pass to the TotoModule constructor.
    """

    def __init__(
        self,
        config
    ) -> None:
        super().__init__()
        # config
        self.config = build_model_config(config)
        model_kwargs = dict(self.config)
        model_kwargs.pop("model_type", None)
        model_kwargs.pop("model_id", None)
        
        # build model
        self.model = TimesBERT(**model_kwargs)


    def load_from_checkpoint(self, checkpoint_path: str | Path) -> None:
        pass
    
    
    def _save_pretrained(self, save_directory: Path) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        with open(save_directory / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

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

        if model_id_path.is_dir():
            config_path = model_id_path / "config.json"
            weight_path = model_id_path / "model.safetensors"
        else:
            raise ValueError(f"Unsupported model_id format: {model_id}. Only local directory paths are supported in this implementation.")
            # config_path = hf_hub_download(
            #     repo_id=model_id,
            #     filename="config.json",
            #     revision=revision,
            #     cache_dir=cache_dir,
            #     force_download=force_download,
            #     local_files_only=local_files_only,
            #     token=token,
            # )
            # weight_path = hf_hub_download(
            #     repo_id=model_id,
            #     filename="model.safetensors",
            #     revision=revision,
            #     cache_dir=cache_dir,
            #     force_download=force_download,
            #     local_files_only=local_files_only,
            #     token=token,
            # )

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 如果 from_pretrained 额外传入参数，可以覆盖 config
        if model_kwargs:
            config.update(model_kwargs)

        model = cls(config)

        state_dict = load_file(weight_path, device=map_location)
        model.load_state_dict(state_dict, strict=strict)

        model.eval()
        return model