from __future__ import annotations

import lightning as L
import torch

from src.data.utils import Timeseries

from .model import Model


class _MaskedReconstructionModule(L.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.lr = float(config["optimization"].get("lr", 2e-4))
        self.weight_decay = float(config["optimization"].get("weight_decay", 1e-2))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _masked_mse_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        point_mask: torch.Tensor,
    ) -> torch.Tensor:
        error = (prediction - target).pow(2).transpose(1, 2)
        return error[point_mask].mean()

    def _shared_step(self, batch: Timeseries, stage: str) -> torch.Tensor:
        x = batch.series
        point_mask = self.model.sample_span_mask(batch_size=x.shape[0], device=x.device)
        prediction = self.model(x, point_mask=point_mask)
        loss = self._masked_mse_loss(prediction, x, point_mask)
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
            point_mask.float().mean(),
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


class ModelForFinetuning(_MaskedReconstructionModule):
    """Masked span reconstruction module used during finetuning."""

    def __init__(self, pretrained_backbone, config: dict) -> None:
        super().__init__(config)
        self.save_hyperparameters(ignore=["pretrained_backbone"])
        self.model = pretrained_backbone if pretrained_backbone is not None else Model(config).model


class ModelForPreTraining(_MaskedReconstructionModule):
    """Same masked span reconstruction objective for optional pretraining."""

    def __init__(self, backbone, config: dict) -> None:
        super().__init__(config)
        self.save_hyperparameters(ignore=["backbone"])
        self.model = backbone


__all__ = ["ModelForFinetuning", "ModelForPreTraining"]
