from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch.nn import functional as F

from src.data.utils import Timeseries

from .timesbert import Model
from .backbone import random_patch_mask


class ModelForFinetuning(L.LightningModule):
    """Full reconstruction finetuning module for TimesBERT."""

    def __init__(
        self,
        pretrained_backbone,
        config
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained_backbone"])
        
        # build backbone
        if pretrained_backbone is not None:
            self.model = pretrained_backbone
        else:
            self.model = Model(config).model
            
        # Training config
        self.lr = float(config["optimization"].get("lr", 2e-4))
        self.weight_decay = float(config["optimization"].get("weight_decay", 1e-2))


    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _point_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(prediction, target)


    def _shared_step(self, batch: Timeseries, stage: str) -> torch.Tensor:
        x = batch.series
        prediction = self(x)
        loss = self._point_loss(prediction, x)
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
    """Masked Patch Modeling pretraining module for TimesBERT."""

    def __init__(
        self,
        backbone,
        config
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.model = backbone
        self.lr = float(config["optimization"].get("lr", 2e-4))
        self.weight_decay = float(config["optimization"].get("weight_decay", 1e-2))
        self.mask_ratio = float(config["trainer"].get("mask_ratio", 0.25))

    def _patch_loss(
        self,
        prediction_patches: torch.Tensor,
        target_patches: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        # mse loss
        error = (prediction_patches - target_patches).pow(2)
        patch_error = error.mean(dim=-1)
        return patch_error[patch_mask].mean()

    def _shared_step(self, batch: Timeseries, stage: str) -> torch.Tensor:
        x = batch.series
        patch_mask = random_patch_mask(
            batch_size=x.shape[0],
            num_features=self.model.num_features,
            patch_count=self.model.patch_count,
            mask_ratio=self.mask_ratio,
            device=x.device,
        )
        prediction_patches = self.model.reconstruct_patches(x, patch_mask=patch_mask)
        target_patches = self.model.patchify(x)
        loss = self._patch_loss(prediction_patches, target_patches, patch_mask)
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
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
