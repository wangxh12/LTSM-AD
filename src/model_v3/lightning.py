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
) -> torch.Tensor:
    if reconstruction.shape != targets.shape:
        raise ValueError("reconstruction and targets must have matching shapes")
    if tuple(valid_mask.shape) != tuple(targets.shape[:2]):
        raise ValueError("valid_mask must match values shape [batch, time]")

    selected = valid_mask.bool()
    if not selected.any():
        raise ValueError("loss mask selects no valid points")
    point_losses = (reconstruction - targets).pow(2).mean(dim=-1)
    return point_losses[selected].mean()


def patch_reconstruction_mse(
    reconstruction_patches: torch.Tensor,
    target_patches: torch.Tensor,
    patch_mask: torch.Tensor,
) -> torch.Tensor:
    if reconstruction_patches.shape != target_patches.shape:
        raise ValueError("reconstruction_patches and target_patches must have matching shapes")
    if tuple(patch_mask.shape) != tuple(target_patches.shape[:3]):
        raise ValueError("patch_mask must match patch shape [batch, features, patch_count]")

    selected = patch_mask.bool()
    if not selected.any():
        raise ValueError("loss mask selects no valid patches")
    patch_losses = (reconstruction_patches - target_patches).pow(2).mean(dim=-1)
    return patch_losses[selected].mean()


def _patch_valid_mask(valid_mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    if valid_mask.ndim != 2:
        raise ValueError(f"valid_mask must have shape [batch, time], got {tuple(valid_mask.shape)}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if valid_mask.shape[-1] % patch_size != 0:
        raise ValueError(
            f"valid_mask time length ({valid_mask.shape[-1]}) must be divisible by patch_size ({patch_size})"
        )
    return valid_mask.bool().unfold(dimension=-1, size=patch_size, step=patch_size).all(dim=-1)


def _random_patch_mask(
    valid_mask: torch.Tensor,
    num_features: int,
    patch_size: int,
    mask_ratio: float,
) -> torch.Tensor:
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
    if num_features <= 0:
        raise ValueError(f"num_features must be positive, got {num_features}")

    patch_valid = _patch_valid_mask(valid_mask, patch_size)
    batch_size, patch_count = patch_valid.shape
    patch_mask = torch.zeros(
        batch_size,
        int(num_features),
        patch_count,
        dtype=torch.bool,
        device=valid_mask.device,
    )

    for row in range(batch_size):
        valid_patch_indices = torch.nonzero(patch_valid[row], as_tuple=False).flatten()
        if len(valid_patch_indices) == 0:
            continue
        mask_count = max(1, math.ceil(len(valid_patch_indices) * mask_ratio))
        mask_count = min(mask_count, len(valid_patch_indices))
        random_scores = torch.rand(int(num_features), len(valid_patch_indices), device=valid_mask.device)
        selected_offsets = random_scores.topk(k=mask_count, dim=-1).indices
        selected_patches = valid_patch_indices[selected_offsets]
        patch_mask[row].scatter_(dim=-1, index=selected_patches, value=True)

    return patch_mask


class _ModelV3LightningModule(L.LightningModule):
    def __init__(self, backbone: nn.Module, config: Mapping[str, Any], objective: Objective) -> None:
        super().__init__()
        self.model = backbone
        self.config = dict(config)
        self.objective = objective
        self.learning_rate = float(config["optimization"].get("lr", 2e-4))
        self.weight_decay = float(config["optimization"].get("weight_decay", 1e-2))
        self.mask_ratio = float(config.get("trainer", {}).get("mask_ratio", 0.25))
        self.patch_size = self._patch_size()
        self.num_features = self._num_features()
        self.save_hyperparameters(ignore=["backbone", "pretrained_backbone"])

    def _patch_size(self) -> int:
        if not hasattr(self.model, "patch_size"):
            raise AttributeError("model_v3 backbone must expose patch_size")
        return int(self.model.patch_size)

    def _num_features(self) -> int:
        if hasattr(self.model, "input_channels"):
            return int(self.model.input_channels)
        data_cfg = self.config.get("data", {})
        target_fields = data_cfg.get("target_fields", self.config.get("features"))
        if not target_fields:
            raise KeyError("Expected data.target_fields or model.input_channels for model_v3")
        return len(target_fields)

    def configure_optimizers(self) -> AdamW:
        return AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def _prepare_values(self, values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return values * valid_mask.unsqueeze(-1).to(dtype=values.dtype)

    def _variate_values(self, values: torch.Tensor) -> torch.Tensor:
        return values.transpose(1, 2).contiguous()

    def _backbone_masks(self, variate_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_padding_mask = torch.ones_like(variate_values, dtype=torch.bool)
        id_mask = torch.zeros_like(variate_values, dtype=torch.long)
        return input_padding_mask, id_mask

    def _forward_backbone(self, values: torch.Tensor, patch_mask: torch.Tensor | None = None) -> torch.Tensor:
        variate_values = self._variate_values(values)
        input_padding_mask, id_mask = self._backbone_masks(variate_values)
        reconstruction = self.model(variate_values, input_padding_mask, id_mask, patch_mask=patch_mask)
        return reconstruction.transpose(1, 2).contiguous()

    def _forward_backbone_patches(self, values: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        variate_values = self._variate_values(values)
        input_padding_mask, id_mask = self._backbone_masks(variate_values)
        return self.model.reconstruct_patches(
            variate_values,
            input_padding_mask,
            id_mask,
            patch_mask=patch_mask,
        )

    def _target_patches(self, values: torch.Tensor) -> torch.Tensor:
        return self.model.patchify(self._variate_values(values))

    def _validate_batch_masks(self, values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        valid_mask = valid_mask.bool()
        if tuple(valid_mask.shape) != tuple(values.shape[:2]):
            raise ValueError("valid_mask must match values shape [batch, time]")
        if values.shape[1] % self.patch_size != 0:
            raise ValueError(
                f"values time length ({values.shape[1]}) must be divisible by patch_size ({self.patch_size})"
            )
        return valid_mask

    def forward(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        patch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if valid_mask is None:
            valid_mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        valid_mask = self._validate_batch_masks(values, valid_mask)

        if patch_mask is not None:
            expected_patch_shape = (values.shape[0], self.num_features, values.shape[1] // self.patch_size)
            if tuple(patch_mask.shape) != expected_patch_shape:
                raise ValueError(
                    "patch_mask must match shape [batch, features, patch_count], "
                    f"expected {expected_patch_shape}, got {tuple(patch_mask.shape)}"
                )
            patch_mask = patch_mask.bool()

        model_values = self._prepare_values(values, valid_mask)
        reconstruction = self._forward_backbone(model_values, patch_mask=patch_mask)
        return reconstruction * valid_mask.unsqueeze(-1).to(dtype=reconstruction.dtype)

    def _shared_step(self, batch: Timeseries, stage: str) -> torch.Tensor:
        values = batch.series
        valid_mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        valid_mask = self._validate_batch_masks(values, valid_mask)

        if self.objective == "pretrain":
            patch_mask = _random_patch_mask(
                valid_mask=valid_mask,
                num_features=values.shape[-1],
                patch_size=self.patch_size,
                mask_ratio=self.mask_ratio,
            )
            patch_valid = _patch_valid_mask(valid_mask, self.patch_size).unsqueeze(1).expand_as(patch_mask)
            patch_mask = patch_mask & patch_valid
            model_values = self._prepare_values(values, valid_mask)
            reconstruction_patches = self._forward_backbone_patches(model_values, patch_mask=patch_mask)
            target_patches = self._target_patches(values)
            loss = patch_reconstruction_mse(reconstruction_patches, target_patches, patch_mask)
            self.log(
                f"{stage}_mask_ratio",
                patch_mask.float().mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=values.shape[0],
            )
        elif self.objective == "finetune":
            reconstruction = self(values, valid_mask=valid_mask)
            loss = reconstruction_mse(reconstruction, values, valid_mask)
        else:
            raise ValueError(f"Unsupported objective: {self.objective}")

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


class ModelForPreTraining(_ModelV3LightningModule):
    def __init__(self, backbone: nn.Module, config: Mapping[str, Any]) -> None:
        super().__init__(backbone=backbone, config=config, objective="pretrain")


class ModelForFinetuning(_ModelV3LightningModule):
    def __init__(self, pretrained_backbone: nn.Module | None, config: Mapping[str, Any]) -> None:
        backbone = pretrained_backbone if pretrained_backbone is not None else Model(dict(config)).model
        super().__init__(backbone=backbone, config=config, objective="finetune")


__all__ = [
    "ModelForFinetuning",
    "ModelForPreTraining",
    "patch_reconstruction_mse",
    "reconstruction_mse",
]
