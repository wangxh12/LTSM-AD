from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = values.float() * torch.rsqrt(values.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(values.dtype)


def rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even")
        frequencies = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inverse_frequencies", frequencies, persistent=False)

    def cos_sin(
        self,
        sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
        frequencies = torch.outer(positions, self.inverse_frequencies.to(device))
        angles = torch.cat((frequencies, frequencies), dim=-1)
        return angles.cos().to(dtype)[None, None], angles.sin().to(dtype)[None, None]

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.cos_sin(queries.shape[-2], queries.device, queries.dtype)
        return queries * cos + rotate_half(queries) * sin, keys * cos + rotate_half(keys) * sin


class TimeSelfAttention(nn.Module):
    """Bidirectional self-attention over time with rotary positions."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float, rope_theta: float) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.qkv_projection = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.output_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, rope_theta)

    def forward(self, values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = values.shape
        qkv = self.qkv_projection(values)
        qkv = qkv.view(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
        queries, keys, attention_values = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        queries, keys = self.rope(queries, keys)

        key_mask = valid_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            queries,
            keys,
            attention_values,
            attn_mask=key_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.hidden_size)
        return self.output_projection(attended)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int, dropout: float) -> None:
        super().__init__()
        self.gate_projection = nn.Linear(hidden_size, ffn_size, bias=False)
        self.up_projection = nn.Linear(hidden_size, ffn_size, bias=False)
        self.down_projection = nn.Linear(ffn_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate_projection(values)) * self.up_projection(values)
        return self.dropout(self.down_projection(hidden))


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_size: int,
        dropout: float,
        rms_norm_eps: float,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(hidden_size, rms_norm_eps)
        self.attention = TimeSelfAttention(hidden_size, num_heads, dropout, rope_theta)
        self.feed_forward_norm = RMSNorm(hidden_size, rms_norm_eps)
        self.feed_forward = SwiGLUFeedForward(hidden_size, ffn_size, dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        mask = valid_mask.unsqueeze(-1).to(values.dtype)
        values = values + self.residual_dropout(self.attention(self.attention_norm(values), valid_mask))
        values = values * mask
        values = values + self.feed_forward(self.feed_forward_norm(values))
        return values * mask


class TimeEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        ffn_size: int,
        dropout: float,
        rms_norm_eps: float,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    ffn_size=ffn_size,
                    dropout=dropout,
                    rms_norm_eps=rms_norm_eps,
                    rope_theta=rope_theta,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_size, rms_norm_eps)

    def forward(self, values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            values = layer(values, valid_mask)
        return self.final_norm(values) * valid_mask.unsqueeze(-1).to(values.dtype)


class PointWiseTokenizer(nn.Module):
    """Map the complete feature vector at each time point to one gated token."""

    def __init__(self, feature_count: int, hidden_size: int) -> None:
        super().__init__()
        self.feature_count = feature_count
        self.hidden_size = hidden_size
        self.value_projection = nn.Linear(feature_count, hidden_size)
        self.gate_projection = nn.Linear(feature_count, hidden_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[-1] != self.feature_count:
            raise ValueError(
                f"Expected values shaped [batch, time, {self.feature_count}], got {tuple(values.shape)}"
            )
        return self.value_projection(values) * F.silu(self.gate_projection(values))



class ReconstructionHead(nn.Module):
    def __init__(self, hidden_size: int, feature_count: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, feature_count)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.projection(tokens)


class ReconstructionModel(nn.Module):
    """Point-wise bidirectional Transformer reconstruction model."""

    def __init__(
        self,
        feature_count: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_size: int = 1024,
        dropout: float = 0.1,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.feature_count = feature_count
        self.hidden_size = hidden_size
        self.tokenizer = PointWiseTokenizer(feature_count, hidden_size)
        self.mask_token = nn.Parameter(torch.empty(hidden_size))
        self.encoder = TimeEncoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_size=ffn_size,
            dropout=dropout,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
        )
        self.head = ReconstructionHead(hidden_size, feature_count)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        point_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid_mask = valid_mask.bool()
        tokens = self.tokenizer(values)

        if point_mask is not None:
            if point_mask.shape != valid_mask.shape:
                raise ValueError("point_mask must match valid_mask shape")
            effective_mask = point_mask.bool() & valid_mask
            tokens = torch.where(effective_mask.unsqueeze(-1), self.mask_token.to(tokens.dtype), tokens)

        tokens = tokens * valid_mask.unsqueeze(-1).to(tokens.dtype)
        reconstruction = self.head(self.encoder(tokens, valid_mask))
        return reconstruction * valid_mask.unsqueeze(-1).to(reconstruction.dtype)