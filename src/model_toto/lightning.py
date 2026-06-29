from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch.nn import functional as F

from src.data.utils import Timeseries

from .backbone import random_patch_mask
from .model import Model


class ModelForFinetuning(L.LightningModule):
    """Full-window reconstruction finetuning module for model_toto."""

    def __init__(self, pretrained_backbone, config: dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained_backbone"])
        self.model = pretrained_backbone if pretrained_backbone is not None else Model(config).model
        self.lr = float(config["optimization"].get("lr", 2e-4))
        self.weight_decay = float(config["optimization"].get("weight_decay", 1e-2))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: Timeseries, stage: str) -> torch.Tensor:
        x = batch.series
        reconstruction = self(x)
        loss = F.mse_loss(reconstruction, x)
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=x.shape[0],
        )
        return loss

    def training_step(self, batch: Timeseries, _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Timeseries, _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")


class ModelForPreTraining(L.LightningModule):
    """Masked patch reconstruction pretraining module for model_toto."""

    def __init__(self, backbone, config: dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.model = backbone
        self.lr = float(config["optimization"].get("lr", 2e-4))
        self.weight_decay = float(config["optimization"].get("weight_decay", 1e-2))
        self.mask_ratio = float(config.get("trainer", {}).get("mask_ratio", 0.25))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _patch_loss(
        self,
        reconstruction_patches: torch.Tensor,
        target_patches: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        patch_errors = (reconstruction_patches - target_patches).pow(2).mean(dim=-1)
        return patch_errors[patch_mask].mean()

    def _shared_step(self, batch: Timeseries, stage: str) -> torch.Tensor:
        x = batch.series
        patch_mask = random_patch_mask(
            batch_size=x.shape[0],
            num_features=self.model.num_features,
            patch_count=self.model.patch_count,
            mask_ratio=self.mask_ratio,
            device=x.device,
        )
        reconstruction_patches = self.model.reconstruct_patches(x, patch_mask=patch_mask)
        target_patches = self.model.patchify(x)
        loss = self._patch_loss(reconstruction_patches, target_patches, patch_mask)
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=x.shape[0],
        )
        self.log(
            f"{stage}_mask_ratio",
            patch_mask.float().mean(),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=x.shape[0],
        )
        return loss

    def training_step(self, batch: Timeseries, _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Timeseries, _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")


__all__ = ["ModelForPreTraining", "ModelForFinetuning"]
