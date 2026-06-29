from __future__ import annotations

import torch
from torch import nn


def rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class TimeAwareRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        theta: float = 10_000.0,
        use_xpos: bool = False,
        cache_if_possible: bool = True,
        seq_before_head_dim: bool = False,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")
        if use_xpos:
            raise ValueError("model_v2 local RoPE does not implement xPos scaling")
        frequencies = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.dim = int(dim)
        self.register_buffer("inverse_frequencies", frequencies, persistent=False)

    def _cos_sin(
        self,
        sequence_length: int,
        seq_pos_offset: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_pos_offset, seq_pos_offset + sequence_length, device=device, dtype=torch.float32)
        frequencies = torch.outer(positions, self.inverse_frequencies.to(device))
        angles = torch.cat((frequencies, frequencies), dim=-1)
        return angles.cos().to(dtype)[None, None], angles.sin().to(dtype)[None, None]

    def rotate_queries_and_keys(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_pos_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._cos_sin(q.shape[-2], seq_pos_offset, q.device, q.dtype)
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


__all__ = ["TimeAwareRotaryEmbedding", "rotate_half"]
