from __future__ import annotations

import math

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name!r}")


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class TotoEncoderLayer(nn.Module):
    """Bidirectional axial encoder over patch time and variate axes."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ffn: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.time_norm = nn.LayerNorm(d_model)
        self.time_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.space_norm = nn.LayerNorm(d_model)
        self.space_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, d_ffn=d_ffn, dropout=dropout, activation=activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, num_features, patch_count, d_model = values.shape

        time_inputs = self.time_norm(values).reshape(batch_size * num_features, patch_count, d_model)
        time_outputs, _ = self.time_attention(
            time_inputs,
            time_inputs,
            time_inputs,
            need_weights=False,
        )
        values = values + self.dropout(time_outputs.reshape(batch_size, num_features, patch_count, d_model))

        space_inputs = (
            self.space_norm(values)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch_size * patch_count, num_features, d_model)
        )
        space_outputs, _ = self.space_attention(
            space_inputs,
            space_inputs,
            space_inputs,
            need_weights=False,
        )
        space_outputs = (
            space_outputs.reshape(batch_size, patch_count, num_features, d_model)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        values = values + self.dropout(space_outputs)
        return values + self.ffn(self.ffn_norm(values))


class TotoReconstructionModel(nn.Module):
    """Toto-style patch encoder for multivariate window reconstruction.

    The public input/output shape is [batch, seq_len, num_features]. Internally,
    each variate is split into fixed-size patches and encoded with bidirectional
    axial attention over patch time and variate axes.
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        patch_len: int = 4,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        d_ffn: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.num_features = int(num_features)
        self.patch_len = int(patch_len)
        self.patch_count = self.seq_len // self.patch_len
        self.d_model = int(d_model)

        self.patch_projection = nn.Linear(self.patch_len, self.d_model)
        self.patch_position_embedding = nn.Parameter(torch.zeros(1, 1, self.patch_count, self.d_model))
        self.variable_embedding = nn.Parameter(torch.zeros(1, self.num_features, 1, self.d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.d_model))
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                TotoEncoderLayer(
                    d_model=self.d_model,
                    num_heads=int(num_heads),
                    d_ffn=int(d_ffn),
                    dropout=float(dropout),
                    activation=activation,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.output_norm = nn.LayerNorm(self.d_model)
        self.reconstruction_head = nn.Linear(self.d_model, self.patch_len)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.patch_position_embedding, std=0.02)
        nn.init.trunc_normal_(self.variable_embedding, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.xavier_uniform_(self.patch_projection.weight)
        nn.init.zeros_(self.patch_projection.bias)
        nn.init.xavier_uniform_(self.reconstruction_head.weight)
        nn.init.zeros_(self.reconstruction_head.bias)

    def patchify(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_features = values.shape
        values = values.transpose(1, 2).contiguous()
        return values.view(batch_size, num_features, self.patch_count, self.patch_len)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch_size, num_features, _, _ = patches.shape
        values = patches.contiguous().view(batch_size, num_features, self.seq_len)
        return values.transpose(1, 2).contiguous()

    def encode(self, values: torch.Tensor, patch_mask: torch.Tensor | None = None) -> torch.Tensor:
        patch_tokens = self.patch_projection(self.patchify(values))
        patch_tokens = patch_tokens + self.patch_position_embedding + self.variable_embedding

        if patch_mask is not None:
            mask_tokens = self.mask_token + self.patch_position_embedding + self.variable_embedding
            patch_tokens = torch.where(patch_mask.unsqueeze(-1), mask_tokens.expand_as(patch_tokens), patch_tokens)

        hidden = self.input_dropout(patch_tokens)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.output_norm(hidden)

    def reconstruct_patches(self, values: torch.Tensor, patch_mask: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encode(values, patch_mask=patch_mask)
        return self.reconstruction_head(encoded)

    def forward(self, values: torch.Tensor, patch_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.unpatchify(self.reconstruct_patches(values, patch_mask=patch_mask))


def random_patch_mask(
    batch_size: int,
    num_features: int,
    patch_count: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
    mask_count = max(1, int(math.ceil(patch_count * mask_ratio)))
    mask_count = min(mask_count, patch_count)
    scores = torch.rand(batch_size, num_features, patch_count, device=device)
    selected = scores.topk(k=mask_count, dim=-1).indices
    mask = torch.zeros(batch_size, num_features, patch_count, dtype=torch.bool, device=device)
    mask.scatter_(dim=-1, index=selected, value=True)
    return mask


__all__ = ["TotoReconstructionModel", "random_patch_mask"]
