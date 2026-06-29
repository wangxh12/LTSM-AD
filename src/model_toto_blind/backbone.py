from __future__ import annotations

import math

import torch
from torch import nn


class BlindPatchReconstructionModel(nn.Module):
    """Reconstruct every patch from all *other* patches in a window.

    Query tokens contain only patch-time and variate identity.  The sole
    attention layer blocks the matching input patch, so a target patch cannot
    pass through an identity path to its reconstruction.  A single layer is
    intentional: stacking cross-patch layers would let a patch leak back to
    itself through another patch token.
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        patch_len: int,
        d_model: int,
        num_heads: int,
        d_ffn: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if seq_len % patch_len != 0:
            raise ValueError(f"seq_len={seq_len} must be divisible by patch_len={patch_len}")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

        self.seq_len = int(seq_len)
        self.num_features = int(num_features)
        self.patch_len = int(patch_len)
        self.patch_count = self.seq_len // self.patch_len
        self.d_model = int(d_model)
        self.token_count = self.num_features * self.patch_count

        self.patch_projection = nn.Linear(self.patch_len, self.d_model)
        self.patch_position_embedding = nn.Parameter(torch.empty(1, self.patch_count, self.d_model))
        self.variable_embedding = nn.Parameter(torch.empty(self.num_features, 1, self.d_model))
        self.mask_token = nn.Parameter(torch.empty(1, 1, self.d_model))
        self.query_norm = nn.LayerNorm(self.d_model)
        self.context_norm = nn.LayerNorm(self.d_model)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.d_model)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, self.d_model),
            nn.Dropout(dropout),
        )
        self.reconstruction_head = nn.Linear(self.d_model, self.patch_len)
        self.register_buffer("self_patch_mask", torch.eye(self.token_count, dtype=torch.bool), persistent=False)
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
        if values.ndim != 3:
            raise ValueError(f"Expected values shaped [batch, seq_len, features], got {tuple(values.shape)}")
        batch_size, seq_len, feature_count = values.shape
        if seq_len != self.seq_len or feature_count != self.num_features:
            raise ValueError(
                f"Expected values shaped [batch, {self.seq_len}, {self.num_features}], got {tuple(values.shape)}"
            )
        return values.transpose(1, 2).contiguous().view(
            batch_size,
            self.num_features,
            self.patch_count,
            self.patch_len,
        )

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        expected_shape = (self.num_features, self.patch_count, self.patch_len)
        if patches.ndim != 4 or tuple(patches.shape[1:]) != expected_shape:
            raise ValueError(f"Expected patches shaped [batch, {expected_shape}], got {tuple(patches.shape)}")
        return patches.contiguous().view(patches.shape[0], self.num_features, self.seq_len).transpose(1, 2).contiguous()

    def _identity_tokens(self) -> torch.Tensor:
        return self.variable_embedding + self.patch_position_embedding

    def _flatten_tokens(self, values: torch.Tensor) -> torch.Tensor:
        return values.reshape(values.shape[0], self.token_count, self.d_model)

    def reconstruct_patches(self, values: torch.Tensor, patch_mask: torch.Tensor | None = None) -> torch.Tensor:
        patches = self.patchify(values)
        identity_tokens = self._identity_tokens()
        context = self.patch_projection(patches) + identity_tokens
        if patch_mask is not None:
            expected_shape = (values.shape[0], self.num_features, self.patch_count)
            if tuple(patch_mask.shape) != expected_shape:
                raise ValueError(f"Expected patch_mask shape {expected_shape}, got {tuple(patch_mask.shape)}")
            masked_context = self.mask_token + identity_tokens
            context = torch.where(patch_mask.unsqueeze(-1), masked_context.expand_as(context), context)

        query = identity_tokens.expand(values.shape[0], -1, -1, -1)
        query = self._flatten_tokens(query)
        context = self._flatten_tokens(context)
        attended, _ = self.cross_attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            attn_mask=self.self_patch_mask,
            need_weights=False,
        )
        hidden = query + attended
        hidden = hidden + self.feed_forward(self.output_norm(hidden))
        reconstructed = self.reconstruction_head(self.output_norm(hidden))
        return reconstructed.view(values.shape[0], self.num_features, self.patch_count, self.patch_len)

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
    scores = torch.rand(batch_size, num_features, patch_count, device=device)
    selected = scores.topk(k=mask_count, dim=-1).indices
    patch_mask = torch.zeros(batch_size, num_features, patch_count, dtype=torch.bool, device=device)
    patch_mask.scatter_(dim=-1, index=selected, value=True)
    return patch_mask


__all__ = ["BlindPatchReconstructionModel", "random_patch_mask"]
