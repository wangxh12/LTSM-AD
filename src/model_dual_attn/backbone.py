from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = float(eps)

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
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
        inverse_frequencies = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inverse_frequencies", inverse_frequencies, persistent=False)

    def _cos_sin(
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
        cos, sin = self._cos_sin(queries.shape[-2], queries.device, queries.dtype)
        return queries * cos + rotate_half(queries) * sin, keys * cos + rotate_half(keys) * sin


class SelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        use_rope: bool,
        rope_theta: float,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.dropout = float(dropout)
        self.qkv_projection = nn.Linear(self.d_model, self.d_model * 3, bias=False)
        self.output_projection = nn.Linear(self.d_model, self.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, rope_theta) if use_rope else None

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = values.shape
        qkv = self.qkv_projection(values)
        qkv = qkv.view(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
        queries, keys, attention_values = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)

        if self.rope is not None:
            queries, keys = self.rope(queries, keys)

        attended = F.scaled_dot_product_attention(
            queries,
            keys,
            attention_values,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.d_model)
        return self.output_projection(attended)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_size: int, dropout: float) -> None:
        super().__init__()
        self.gate_projection = nn.Linear(d_model, ffn_size, bias=False)
        self.up_projection = nn.Linear(d_model, ffn_size, bias=False)
        self.down_projection = nn.Linear(ffn_size, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate_projection(values)) * self.up_projection(values)
        return self.dropout(self.down_projection(hidden))


class DualAttentionEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_size: int,
        dropout: float,
        rms_norm_eps: float,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.time_norm = RMSNorm(d_model, rms_norm_eps)
        self.time_attention = SelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=True,
            rope_theta=rope_theta,
        )
        self.variable_norm = RMSNorm(d_model, rms_norm_eps)
        self.variable_attention = SelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=False,
            rope_theta=rope_theta,
        )
        self.ffn_norm = RMSNorm(d_model, rms_norm_eps)
        self.feed_forward = SwiGLUFeedForward(d_model=d_model, ffn_size=ffn_size, dropout=dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, num_features, seq_len, d_model = values.shape

        time_inputs = self.time_norm(values).reshape(batch_size * num_features, seq_len, d_model)
        time_outputs = self.time_attention(time_inputs)
        values = values + self.residual_dropout(time_outputs.reshape(batch_size, num_features, seq_len, d_model))

        variable_inputs = (
            self.variable_norm(values)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch_size * seq_len, num_features, d_model)
        )
        variable_outputs = self.variable_attention(variable_inputs)
        variable_outputs = (
            variable_outputs.reshape(batch_size, seq_len, num_features, d_model)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        values = values + self.residual_dropout(variable_outputs)
        return values + self.feed_forward(self.ffn_norm(values))


class PointDualAttentionReconstructionModel(nn.Module):
    """Point-level axial attention reconstruction model.

    Public input and output shape is [batch, seq_len, num_features].
    Internally each scalar point becomes a token shaped [batch, features, time, d_model].
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        d_model: int = 16,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_size: int = 64,
        dropout: float = 0.1,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10_000.0,
        mask_span_len: int = 4,
        mask_ratio: float = 0.30,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.num_features = int(num_features)
        self.d_model = int(d_model)
        self.mask_span_len = int(mask_span_len)
        self.mask_ratio = float(mask_ratio)

        if self.seq_len % self.mask_span_len != 0:
            raise ValueError(f"seq_len={self.seq_len} must be divisible by mask_span_len={self.mask_span_len}")
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {self.mask_ratio}")
        if self.d_model % int(num_heads) != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by num_heads={num_heads}")

        self.value_projection = nn.Linear(1, self.d_model)
        self.gate_projection = nn.Linear(1, self.d_model)
        self.mask_token = nn.Parameter(torch.empty(self.d_model))
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                DualAttentionEncoderLayer(
                    d_model=self.d_model,
                    num_heads=int(num_heads),
                    ffn_size=int(ffn_size),
                    dropout=float(dropout),
                    rms_norm_eps=float(rms_norm_eps),
                    rope_theta=float(rope_theta),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.output_norm = RMSNorm(self.d_model, rms_norm_eps)
        self.reconstruction_head = nn.Linear(self.d_model, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.value_projection.weight)
        nn.init.zeros_(self.value_projection.bias)
        nn.init.xavier_uniform_(self.gate_projection.weight)
        nn.init.zeros_(self.gate_projection.bias)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.xavier_uniform_(self.reconstruction_head.weight)
        nn.init.zeros_(self.reconstruction_head.bias)

    @property
    def span_count(self) -> int:
        return self.seq_len // self.mask_span_len

    def sample_span_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return random_span_mask(
            batch_size=batch_size,
            num_features=self.num_features,
            seq_len=self.seq_len,
            span_len=self.mask_span_len,
            mask_ratio=self.mask_ratio,
            device=device,
        )

    def point_embed(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_features = values.shape
        if seq_len != self.seq_len or num_features != self.num_features:
            raise ValueError(
                f"Expected input shape [batch, {self.seq_len}, {self.num_features}], got {tuple(values.shape)}"
            )
        points = values.transpose(1, 2).unsqueeze(-1).contiguous()
        return F.silu(self.gate_projection(points)) * self.value_projection(points)

    def encode(self, values: torch.Tensor, point_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.point_embed(values)
        if point_mask is not None:
            expected_shape = (values.shape[0], self.num_features, self.seq_len)
            if tuple(point_mask.shape) != expected_shape:
                raise ValueError(f"Expected point_mask shape {expected_shape}, got {tuple(point_mask.shape)}")
            mask_token = self.mask_token.to(dtype=hidden.dtype).view(1, 1, 1, self.d_model)
            hidden = torch.where(point_mask.unsqueeze(-1), mask_token.expand_as(hidden), hidden)

        hidden = self.input_dropout(hidden)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.output_norm(hidden)

    def forward(self, values: torch.Tensor, point_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.encode(values, point_mask=point_mask)
        reconstruction = self.reconstruction_head(hidden).squeeze(-1)
        return reconstruction.transpose(1, 2).contiguous()


def random_span_mask(
    batch_size: int,
    num_features: int,
    seq_len: int,
    span_len: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    if seq_len % span_len != 0:
        raise ValueError(f"seq_len={seq_len} must be divisible by span_len={span_len}")
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")

    span_count = seq_len // span_len
    masked_span_count = max(1, int(math.ceil(span_count * mask_ratio)))
    masked_span_count = min(masked_span_count, span_count)
    scores = torch.rand(batch_size, num_features, span_count, device=device)
    selected = scores.topk(k=masked_span_count, dim=-1).indices
    span_mask = torch.zeros(batch_size, num_features, span_count, dtype=torch.bool, device=device)
    span_mask.scatter_(dim=-1, index=selected, value=True)
    return span_mask.repeat_interleave(span_len, dim=-1)


__all__ = [
    "DualAttentionEncoderLayer",
    "PointDualAttentionReconstructionModel",
    "RMSNorm",
    "RotaryEmbedding",
    "SelfAttention",
    "SwiGLUFeedForward",
    "random_span_mask",
]
