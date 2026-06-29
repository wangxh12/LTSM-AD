from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch.nn import functional as F

from src.data.utils import Timeseries

from .transformer import ReconstructionTransformer


class ReconstructionLitModule(L.LightningModule):
    """Lightning module for point-wise reconstruction pretraining and finetuning."""

    def __init__(
        self,
        model: dict[str, Any],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        loss: str = "mae",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = ReconstructionTransformer(**model)
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_name = loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_name == "mse":
            return F.mse_loss(prediction, target)
        if self.loss_name == "mae":
            return F.l1_loss(prediction, target)
        raise ValueError(f"Unsupported loss: {self.loss_name}")

    def _shared_step(self, batch: Timeseries, stage: str) -> torch.Tensor:
        x = batch.series
        prediction = self(x)
        loss = self._loss(prediction, x)
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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class PretrainLitModule(ReconstructionLitModule):
    """Masked reconstruction module for pretraining.

    The random mask is generated independently for each sample and each variable.
    Each selected mask unit covers consecutive timesteps along the time axis.
    """

    def __init__(
        self,
        model: dict[str, Any],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        loss: str = "mae",
        mask_ratio: float = 0.25,
        mask_unit_size: int = 4,
    ) -> None:
        super().__init__(model=model, lr=lr, weight_decay=weight_decay, loss=loss)
        self.save_hyperparameters()
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
        if mask_unit_size <= 0:
            raise ValueError(f"mask_unit_size must be positive, got {mask_unit_size}")
        self.mask_ratio = float(mask_ratio)
        self.mask_unit_size = int(mask_unit_size)
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, int(model["num_features"])))

    def _random_block_mask(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_features = x.shape
        block_size = min(self.mask_unit_size, seq_len)
        num_units = seq_len // block_size
        if num_units == 0:
            raise ValueError(f"seq_len={seq_len} is too short for mask_unit_size={self.mask_unit_size}")

        num_mask_units = max(1, round(num_units * self.mask_ratio))
        num_mask_units = min(num_mask_units, num_units)
        random_scores = torch.rand(batch_size, num_features, num_units, device=x.device)
        selected_units = random_scores.topk(k=num_mask_units, dim=-1).indices
        unit_mask = torch.zeros(batch_size, num_features, num_units, dtype=torch.bool, device=x.device)
        unit_mask.scatter_(dim=-1, index=selected_units, value=True)

        mask = torch.zeros(batch_size, seq_len, num_features, dtype=torch.bool, device=x.device)
        for unit_index in range(num_units):
            start = unit_index * block_size
            end = start + block_size
            mask[:, start:end, :] = unit_mask[:, :, unit_index].unsqueeze(1)
        return mask

    def _masked_loss(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.loss_name == "mse":
            error = (prediction - target).pow(2)
        elif self.loss_name == "mae":
            error = (prediction - target).abs()
        else:
            raise ValueError(f"Unsupported loss: {self.loss_name}")
        return error[mask].mean()

    def _shared_step(self, batch: Timeseries, stage: str) -> torch.Tensor:
        x = batch.series
        mask = self._random_block_mask(x)
        masked_x = torch.where(mask, self.mask_token.to(dtype=x.dtype, device=x.device), x)
        prediction = self(masked_x)
        loss = self._masked_loss(prediction, x, mask)
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
            mask.float().mean(),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=x.shape[0],
        )
        return loss
