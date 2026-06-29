from __future__ import annotations

import math
from typing import Any, Literal, Mapping

import lightning as L
import torch
from torch.optim import AdamW

from .backbone import ITransformerReconstructionModel

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


def _random_point_mask(valid_mask: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
    valid_mask = valid_mask.bool()
    point_mask = torch.zeros_like(valid_mask)
    for row in range(valid_mask.shape[0]):
        valid_indices = torch.nonzero(valid_mask[row], as_tuple=False).flatten()
        if len(valid_indices) == 0:
            continue
        mask_count = max(1, math.ceil(len(valid_indices) * mask_ratio))
        mask_count = min(mask_count, len(valid_indices))
        selected = valid_indices[torch.randperm(len(valid_indices), device=valid_mask.device)[:mask_count]]
        point_mask[row, selected] = True
    return point_mask


class _ITransformerLightningModule(L.LightningModule):
    def __init__(
        self,
        backbone: ITransformerReconstructionModel,
        config: Mapping[str, Any],
        objective: Objective,
    ) -> None:
        super().__init__()
        self.model = backbone
        self.config = dict(config)
        self.objective = objective
        self.learning_rate = float(config["optimization"].get("lr", 2e-4))
        self.weight_decay = float(config["optimization"].get("weight_decay", 1e-2))
        self.mask_ratio = float(config.get("trainer", {}).get("mask_ratio", 0.25))
        self.save_hyperparameters(ignore=["backbone", "pretrained_backbone"])

    def forward(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        point_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(values, valid_mask=valid_mask, point_mask=point_mask)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        values = batch["values"] if "values" in batch else batch["x"]
        valid_mask = batch.get("valid_mask")
        if valid_mask is None:
            valid_mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)

        point_mask = batch.get("point_mask")
        if self.objective == "pretrain" and point_mask is None:
            point_mask = _random_point_mask(valid_mask, self.mask_ratio)
        if self.objective == "finetune":
            point_mask = None

        reconstruction = self(values, valid_mask=valid_mask, point_mask=point_mask)
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


class ModelForPreTraining(_ITransformerLightningModule):
    def __init__(self, backbone: ITransformerReconstructionModel, config: Mapping[str, Any]) -> None:
        super().__init__(backbone=backbone, config=config, objective="pretrain")


class ModelForFinetuning(_ITransformerLightningModule):
    def __init__(self, pretrained_backbone: ITransformerReconstructionModel, config: Mapping[str, Any]) -> None:
        super().__init__(backbone=pretrained_backbone, config=config, objective="finetune")


__all__ = ["ModelForFinetuning", "ModelForPreTraining", "reconstruction_mse"]
