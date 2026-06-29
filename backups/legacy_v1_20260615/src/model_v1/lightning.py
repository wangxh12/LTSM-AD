from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import lightning as L
import torch
from torch.optim import AdamW

from .transformer import ReconstructionModel

Objective = Literal["pretrain", "finetune"]


def reconstruction_mse(
    reconstruction: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    point_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if reconstruction.shape != targets.shape:
        raise ValueError("reconstruction and targets must have matching shapes")
    selected = valid_mask.bool()
    if point_mask is not None:
        selected = selected & point_mask.bool()
    if not selected.any():
        raise ValueError("loss mask selects no valid points")
    point_losses = (reconstruction - targets).pow(2).mean(dim=-1)
    return point_losses[selected].mean()


class ReconstructionLightningModule(L.LightningModule):
    def __init__(
        self,
        feature_columns: Sequence[str],
        model_config: Mapping[str, Any],
        objective: Objective,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.feature_columns = tuple(feature_columns)
        self.model_config = dict(model_config)
        self.objective = objective
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = ReconstructionModel(feature_count=len(self.feature_columns), **self.model_config)
        self.save_hyperparameters(
            {
                "feature_columns": list(self.feature_columns),
                "model_config": self.model_config,
                "objective": self.objective,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
            }
        )

    def forward(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        point_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if valid_mask is None:
            valid_mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        return self.model(values, valid_mask, point_mask)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        values = batch["values"] if "values" in batch else batch["x"]
        valid_mask = batch.get("valid_mask")
        if valid_mask is None:
            valid_mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        point_mask = batch.get("point_mask") if self.objective == "pretrain" else None
        reconstruction = self(values, valid_mask, point_mask)
        loss = reconstruction_mse(reconstruction, values, valid_mask, point_mask)
        self.log(
            f"{stage}_loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=values.shape[0],
        )
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self) -> AdamW:
        return AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


def load_checkpoint_module(path: str | Path) -> ReconstructionLightningModule:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "hyper_parameters" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError(f"Not a compatible Lightning checkpoint: {path}")
    hyperparameters = checkpoint["hyper_parameters"]
    module = ReconstructionLightningModule(**hyperparameters)
    if module.objective not in ("pretrain", "finetune"):
        raise ValueError(f"Unsupported checkpoint objective: {module.objective}")
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    return module


def validate_checkpoint_features(module: ReconstructionLightningModule, expected_features: Sequence[str]) -> None:
    if tuple(expected_features) != module.feature_columns:
        raise ValueError(
            f"Checkpoint feature order mismatch: expected {list(expected_features)}, "
            f"checkpoint has {list(module.feature_columns)}"
        )


def initialize_from_pretrained(
    module: ReconstructionLightningModule,
    checkpoint_path: str | Path,
) -> None:
    pretrained = load_checkpoint_module(checkpoint_path)
    if pretrained.objective != "pretrain":
        raise ValueError("pretrained_checkpoint must contain a pretraining module")
    validate_checkpoint_features(pretrained, module.feature_columns)
    if pretrained.model_config != module.model_config:
        raise ValueError(
            f"Pretrained model config mismatch: expected {module.model_config}, checkpoint has {pretrained.model_config}"
        )
    module.model.load_state_dict(pretrained.model.state_dict(), strict=True)
