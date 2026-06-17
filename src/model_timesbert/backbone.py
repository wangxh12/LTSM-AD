from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file, save_file


class TimesBERT(nn.Module):
    """TimesBERT-style encoder for multivariate reconstruction.

    A multivariate window is treated as a document: each variate is split into
    patch tokens, variates are separated by a shared [SEP] token, and a global
    [CLS] token is prepended before bidirectional Transformer encoding.
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        patch_len: int = 4,
        d_model: int = 128,
        num_layers: int = 4,
        n_heads: int = 4,
        d_ffn: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        if seq_len % patch_len != 0:
            raise ValueError(f"seq_len={seq_len} must be divisible by patch_len={patch_len}")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.seq_len = int(seq_len)
        self.num_features = int(num_features)
        self.patch_len = int(patch_len)
        self.patch_count = self.seq_len // self.patch_len
        self.d_model = int(d_model)
        self.config = {
            "model_type": "timesbert",
            "seq_len": self.seq_len,
            "num_features": self.num_features,
            "patch_len": self.patch_len,
            "d_model": self.d_model,
            "num_layers": int(num_layers),
            "n_heads": int(n_heads),
            "d_ffn": int(d_ffn),
            "dropout": float(dropout),
            "activation": activation,
            "norm_first": bool(norm_first),
        }

        self.patch_projection = nn.Linear(self.patch_len, self.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, 1, self.patch_count, self.d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, 1, self.d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(self.d_model)
        self.reconstruction_head = nn.Linear(self.d_model, self.patch_len)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    # @classmethod
    # def from_config(cls, config: dict) -> "TimesBERTModel":
    #     model_config = dict(config)
    #     model_config.pop("model_type", None)
    #     model_config.pop("model_id", None)
    #     return cls(**model_config)


    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.sep_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.xavier_uniform_(self.patch_projection.weight)
        nn.init.zeros_(self.patch_projection.bias)
        nn.init.xavier_uniform_(self.reconstruction_head.weight)
        nn.init.zeros_(self.reconstruction_head.bias)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [batch, seq_len, features], got {tuple(x.shape)}")
        batch_size, seq_len, num_features = x.shape
        if seq_len != self.seq_len or num_features != self.num_features:
            raise ValueError(
                f"Expected input shape [batch, {self.seq_len}, {self.num_features}], got {tuple(x.shape)}"
            )
        values = x.transpose(1, 2).contiguous()
        return values.view(batch_size, num_features, self.patch_count, self.patch_len)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch_size, num_features, patch_count, patch_len = patches.shape
        if num_features != self.num_features or patch_count != self.patch_count or patch_len != self.patch_len:
            raise ValueError(f"Unexpected patch tensor shape: {tuple(patches.shape)}")
        values = patches.contiguous().view(batch_size, num_features, self.seq_len)
        return values.transpose(1, 2).contiguous()

    def encode(self, x: torch.Tensor, patch_mask: torch.Tensor | None = None) -> torch.Tensor:
        patches = self.patchify(x)
        patch_tokens = self.patch_projection(patches)
        patch_tokens = patch_tokens + self.position_embedding

        if patch_mask is not None:
            expected_mask_shape = (x.shape[0], self.num_features, self.patch_count)
            if tuple(patch_mask.shape) != expected_mask_shape:
                raise ValueError(f"Expected patch_mask shape {expected_mask_shape}, got {tuple(patch_mask.shape)}")
            mask_tokens = self.mask_token + self.position_embedding
            patch_tokens = torch.where(patch_mask.unsqueeze(-1), mask_tokens.expand_as(patch_tokens), patch_tokens)

        patch_tokens = self.dropout(patch_tokens)
        sep_tokens = self.sep_token.expand(x.shape[0], self.num_features, 1, self.d_model)
        variate_sentences = torch.cat([patch_tokens, sep_tokens], dim=2)
        document_tokens = variate_sentences.reshape(x.shape[0], self.num_features * (self.patch_count + 1), self.d_model)
        cls_tokens = self.cls_token.expand(x.shape[0], 1, self.d_model)
        tokens = torch.cat([cls_tokens, document_tokens], dim=1)

        encoded = self.encoder(tokens)
        encoded = self.output_norm(encoded)
        encoded_document = encoded[:, 1:].reshape(x.shape[0], self.num_features, self.patch_count + 1, self.d_model)
        return encoded_document[:, :, : self.patch_count]

    def reconstruct_patches(self, x: torch.Tensor, patch_mask: torch.Tensor | None = None) -> torch.Tensor:
        encoded_patch_tokens = self.encode(x, patch_mask=patch_mask)
        return self.reconstruction_head(encoded_patch_tokens)

    def forward(self, x: torch.Tensor, patch_mask: torch.Tensor | None = None) -> torch.Tensor:
        reconstructed_patches = self.reconstruct_patches(x, patch_mask=patch_mask)
        return self.unpatchify(reconstructed_patches)


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
