from __future__ import annotations

import torch
from torch import nn

from src.model.embedding import PointWiseEmbedding


class TemporalChannelEncoderLayer(nn.Module):
    """Applies temporal attention per feature, then channel attention per timestep."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.channel_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(d_model)
        self.channel_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_features, d_model = x.shape

        temporal_x = x.permute(0, 2, 1, 3).reshape(batch_size * num_features, seq_len, d_model)
        temporal_out, _ = self.temporal_attn(temporal_x, temporal_x, temporal_x, need_weights=False)
        temporal_x = self.temporal_norm(temporal_x + self.dropout(temporal_out))
        x = temporal_x.reshape(batch_size, num_features, seq_len, d_model).permute(0, 2, 1, 3)

        channel_x = x.reshape(batch_size * seq_len, num_features, d_model)
        channel_out, _ = self.channel_attn(channel_x, channel_x, channel_x, need_weights=False)
        channel_x = self.channel_norm(channel_x + self.dropout(channel_out))
        x = channel_x.reshape(batch_size, seq_len, num_features, d_model)

        ffn_out = self.ffn(x)
        return self.ffn_norm(x + self.dropout(ffn_out))


class ReconstructionTransformer(nn.Module):
    """Small point-wise token reconstruction encoder for UAV anomaly detection."""

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        d_ffn: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.tokenizer = PointWiseEmbedding(d_model)
        self.encoder = nn.ModuleList(
            [
                TemporalChannelEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ffn=d_ffn,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        # self.reconstruction_head = nn.Linear(d_model, 1)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x).transpose(1, 2)
        for layer in self.encoder:
            tokens = layer(tokens)
        return self.reconstruction_head(tokens).squeeze(-1)
