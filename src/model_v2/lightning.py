from __future__ import annotations

import math
from typing import Any, Literal, Mapping

import lightning as L
import torch
from torch import nn
from torch.optim import AdamW

from src.data.utils import Timeseries

from .model import Model

Objective = Literal["pretrain", "finetune"]


def reconstruction_mse(
    reconstruction: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    point_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if reconstruction.shape != targets.shape:
        raise ValueError("reconstruction and targets must have matching shapes")
    if tuple(valid_mask.shape) != tuple(targets.shape[:2]):
        raise ValueError("valid_mask must match values shape [batch, time]")

    selected = valid_mask.bool().unsqueeze(-1).expand_as(targets)
    if point_mask is not None:
        if tuple(point_mask.shape) != tuple(targets.shape):
            raise ValueError("point_mask must match values shape [batch, time, features]")
        selected = selected & point_mask.bool()
    if not selected.any():
        raise ValueError("loss mask selects no valid points")
    point_losses = (reconstruction - targets).pow(2)
    return point_losses[selected].mean()


def _random_point_mask(valid_mask: torch.Tensor, num_features: int, mask_ratio: float) -> torch.Tensor:
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
    valid_mask = valid_mask.bool()
    point_mask = torch.zeros(
        (*valid_mask.shape, int(num_features)),
        dtype=torch.bool,
        device=valid_mask.device,
    )
    for row in range(valid_mask.shape[0]):
        valid_indices = torch.nonzero(
            valid_mask[row].unsqueeze(-1).expand(-1, int(num_features)),
            as_tuple=False,
        )
        if len(valid_indices) == 0:
            continue
        mask_count = max(1, math.ceil(len(valid_indices) * mask_ratio))
        mask_count = min(mask_count, len(valid_indices))
        selected = valid_indices[torch.randperm(len(valid_indices), device=valid_mask.device)[:mask_count]]
        point_mask[row, selected[:, 0], selected[:, 1]] = True
    return point_mask


class _ModelV2LightningModule(L.LightningModule):
    def __init__(self, backbone: nn.Module, config: Mapping[str, Any], objective: Objective) -> None:
        super().__init__()
        self.model = backbone
        self.config = dict(config)
        self.objective = objective
        self.learning_rate = float(config["optimization"].get("lr", 2e-4))
        self.weight_decay = float(config["optimization"].get("weight_decay", 1e-2))
        self.mask_ratio = float(config.get("trainer", {}).get("mask_ratio", 0.25))
        self.save_hyperparameters(ignore=["backbone", "pretrained_backbone"])

    def configure_optimizers(self) -> AdamW:
        return AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def _forward_backbone(self, values: torch.Tensor, point_mask: torch.Tensor | None = None) -> torch.Tensor:
        # transpose to Toto-style model tensor
        variate_values = values.transpose(1, 2).contiguous() # [batch, variate, time_steps]
        input_padding_mask = torch.ones_like(variate_values, dtype=torch.bool)
        id_mask = torch.zeros_like(variate_values, dtype=torch.long)
        variate_point_mask = point_mask.transpose(1, 2).contiguous() if point_mask is not None else None
        reconstruction = self.model(variate_values, input_padding_mask, id_mask, point_mask=variate_point_mask)
        return reconstruction.transpose(1, 2).contiguous()

    def forward(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        point_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if valid_mask is None:
            valid_mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        valid_mask = valid_mask.bool()
        if tuple(valid_mask.shape) != tuple(values.shape[:2]):
            raise ValueError("valid_mask must match values shape [batch, time]")

        model_values = values * valid_mask.unsqueeze(-1).to(dtype=values.dtype)
        if self.objective == "pretrain" and point_mask is not None:
            if tuple(point_mask.shape) != tuple(values.shape):
                raise ValueError("point_mask must match values shape [batch, time, features]")
            point_mask = point_mask.bool() & valid_mask.unsqueeze(-1)
        else:
            point_mask = None

        reconstruction = self._forward_backbone(model_values, point_mask=point_mask)
        return reconstruction * valid_mask.unsqueeze(-1).to(dtype=reconstruction.dtype)

    def _shared_step(self, batch: Timeseries, stage: str) -> torch.Tensor:
        values = batch.series
        valid_mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)

        point_mask = None
        if self.objective == "pretrain" and point_mask is None:
            point_mask = _random_point_mask(valid_mask, values.shape[-1], self.mask_ratio)
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

    def training_step(self, batch: Timeseries, _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Timeseries, _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")


class ModelForPreTraining(_ModelV2LightningModule):
    def __init__(self, backbone: nn.Module, config: Mapping[str, Any]) -> None:
        super().__init__(backbone=backbone, config=config, objective="pretrain")


class ModelForFinetuning(_ModelV2LightningModule):
    def __init__(self, pretrained_backbone: nn.Module | None, config: Mapping[str, Any]) -> None:
        backbone = pretrained_backbone if pretrained_backbone is not None else Model(dict(config)).model
        super().__init__(backbone=backbone, config=config, objective="finetune")


__all__ = ["ModelForFinetuning", "ModelForPreTraining", "reconstruction_mse"]
