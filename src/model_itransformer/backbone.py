from __future__ import annotations

import torch
from torch import nn


class ITransformerReconstructionModel(nn.Module):
    """iTransformer-style reconstruction backbone.

    Each variable is treated as a token. The full temporal trajectory of one
    variable is projected into a token, variable tokens attend to each other,
    and a reconstruction head maps every variable token back to a time series.
    """

    def __init__(
        self,
        seq_len: int,
        feature_count: int,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ffn: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.seq_len = int(seq_len)
        self.feature_count = int(feature_count)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.d_ffn = int(d_ffn)
        self.dropout = float(dropout)
        self.activation = activation
        self.norm_first = bool(norm_first)

        self.mask_token = nn.Parameter(torch.empty(self.feature_count))
        self.input_projection = nn.Linear(self.seq_len, self.d_model)
        self.variable_embedding = nn.Parameter(torch.zeros(1, self.feature_count, self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_ffn,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            norm_first=self.norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.output_norm = nn.LayerNorm(self.d_model)
        self.reconstruction_head = nn.Linear(self.d_model, self.seq_len)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.variable_embedding, std=0.02)

    def forward(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        point_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError(f"Expected values shape [batch, time, features], got {tuple(values.shape)}")
        if values.shape[1] != self.seq_len or values.shape[2] != self.feature_count:
            raise ValueError(
                f"Expected values shape [batch, {self.seq_len}, {self.feature_count}], got {tuple(values.shape)}"
            )

        if valid_mask is None:
            valid_mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        valid_mask = valid_mask.bool()
        if tuple(valid_mask.shape) != tuple(values.shape[:2]):
            raise ValueError("valid_mask must match values shape [batch, time]")

        inputs = values
        if point_mask is not None:
            if tuple(point_mask.shape) != tuple(valid_mask.shape):
                raise ValueError("point_mask must match values shape [batch, time]")
            effective_mask = point_mask.bool() & valid_mask
            mask_token = self.mask_token.to(dtype=values.dtype, device=values.device).view(1, 1, -1)
            inputs = torch.where(effective_mask.unsqueeze(-1), mask_token, inputs)

        inputs = inputs * valid_mask.unsqueeze(-1).to(dtype=inputs.dtype)
        variable_tokens = self.input_projection(inputs.transpose(1, 2))
        variable_tokens = variable_tokens + self.variable_embedding.to(dtype=variable_tokens.dtype)
        encoded = self.encoder(variable_tokens)
        encoded = self.output_norm(encoded)
        reconstruction = self.reconstruction_head(encoded).transpose(1, 2).contiguous()
        return reconstruction * valid_mask.unsqueeze(-1).to(dtype=reconstruction.dtype)


__all__ = ["ITransformerReconstructionModel"]
