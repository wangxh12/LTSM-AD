from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch.nn import functional as F

from src.data.utils import Timeseries

from .backbone import BlindPatchReconstructionModel, random_patch_mask
from .model import Model


class _TotoBlindModule(L.LightningModule):
    def __init__(self, backbone: BlindPatchReconstructionModel, config: dict[str, Any], pretraining: bool) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.model = backbone
        self.pretraining = pretraining
        self.lr = float(config["optimization"].get("lr", 2e-4))
        self.weight_decay = float(config["optimization"].get("weight_decay", 1e-2))
        self.mask_ratio = float(config.get("trainer", {}).get("mask_ratio", 0.25))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.model(values)

    def _loss(self, values: torch.Tensor) -> torch.Tensor:
        if not self.pretraining:
            return F.mse_loss(self(values), values)
        patch_mask = random_patch_mask(
            batch_size=values.shape[0],
            num_features=self.model.num_features,
            patch_count=self.model.patch_count,
            mask_ratio=self.mask_ratio,
            device=values.device,
        )
        target_patches = self.model.patchify(values)
        reconstructed_patches = self.model.reconstruct_patches(values, patch_mask=patch_mask)
        patch_loss = (reconstructed_patches - target_patches).pow(2).mean(dim=-1)
        return patch_loss[patch_mask].mean()

    def _shared_step(self, batch: Timeseries, stage: str) -> torch.Tensor:
        loss = self._loss(batch.series)
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=batch.series.shape[0],
        )
        return loss

    def training_step(self, batch: Timeseries, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Timeseries, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")


class ModelForPreTraining(_TotoBlindModule):
    def __init__(self, backbone: BlindPatchReconstructionModel, config: dict[str, Any]) -> None:
        super().__init__(backbone, config, pretraining=True)


class ModelForFinetuning(_TotoBlindModule):
    def __init__(self, pretrained_backbone: BlindPatchReconstructionModel | None, config: dict[str, Any]) -> None:
        backbone = pretrained_backbone if pretrained_backbone is not None else Model(config).model
        super().__init__(backbone, config, pretraining=False)


__all__ = ["ModelForPreTraining", "ModelForFinetuning"]
