# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

import torch
from jaxtyping import Float, Int, Num


def patchify_id_mask(
    id_mask: Int[torch.Tensor, "batch variate time_steps"], patch_size: int
) -> Int[torch.Tensor, "batch variate seq_len"]:
    if id_mask.ndim != 3:
        raise ValueError(f"id_mask must have shape [batch, variate, time_steps], got {tuple(id_mask.shape)}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if id_mask.shape[-1] % patch_size != 0:
        raise ValueError(
            f"id_mask time length ({id_mask.shape[-1]}) must be divisible by patch_size ({patch_size})"
        )
    patched_id_mask = id_mask.unfold(dimension=-1, size=patch_size, step=patch_size)
    patched_id_mask_min = patched_id_mask.min(-1).values
    patched_id_mask_max = patched_id_mask.max(-1).values
    assert torch.eq(patched_id_mask_min, patched_id_mask_max).all(), "Patches cannot span multiple datasets"
    return patched_id_mask_min


class PatchEmbedding(torch.nn.Module):
    """
    Multivariate time series patch embedding.
    Patchifies each variate separately.
    """

    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.projection = torch.nn.Linear(self.patch_size, self.embed_dim)

    def _patchify(
        self, x: Num[torch.Tensor, "batch variate time_steps"]
    ) -> Num[torch.Tensor, "batch variate seq_len patch_size"]:
        return x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)

    def forward(
        self,
        x: Float[torch.Tensor, "batch #variate time_steps"],
        id_mask: Int[torch.Tensor, "batch variate time_steps"],
    ) -> tuple[
        Float[torch.Tensor, "batch variate seq_len embed_dim"],
        Int[torch.Tensor, "batch variate seq_len"],
    ]:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [batch, variate, time_steps], got {tuple(x.shape)}")
        if tuple(id_mask.shape) != tuple(x.shape):
            raise ValueError("id_mask must match x shape [batch, variate, time_steps]")
        if x.shape[-1] % self.patch_size != 0:
            raise ValueError(
                f"Series length ({x.shape[-1]}) must be divisible by patch_size ({self.patch_size})"
            )
        x_patched: Float[torch.Tensor, "batch variate seq_len patch_size"] = self._patchify(x)
        return self.projection(x_patched), patchify_id_mask(id_mask, self.patch_size)
