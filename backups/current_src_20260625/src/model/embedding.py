import torch
from torch import nn
from torch.nn import functional as F

class PointWiseTokenizer(nn.Module):
    """Turns each scalar value at each time/feature position into a token."""

    def __init__(self, seq_len: int, num_features: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.value_projection = nn.Linear(1, d_model)
        self.feature_embedding = nn.Embedding(num_features, d_model)
        self.time_embedding = nn.Embedding(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input shape [batch, seq_len, features], got {tuple(x.shape)}")
        batch_size, seq_len, num_features = x.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}")
        if num_features != self.num_features:
            raise ValueError(f"Expected num_features={self.num_features}, got {num_features}")

        tokens = self.value_projection(x.unsqueeze(-1))
        feature_ids = torch.arange(num_features, device=x.device)
        time_ids = torch.arange(seq_len, device=x.device)
        feature_emb = self.feature_embedding(feature_ids).view(1, 1, num_features, -1)
        time_emb = self.time_embedding(time_ids).view(1, seq_len, 1, -1)
        return self.dropout(tokens + feature_emb + time_emb)

class PointWiseEmbedding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.emb_layer = nn.Linear(1, d_model)
        self.gate_layer = nn.Linear(1, d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        scalar_values = values.transpose(1, 2).unsqueeze(-1)
        return self.emb_layer(scalar_values) * F.silu(self.gate_layer(scalar_values))
