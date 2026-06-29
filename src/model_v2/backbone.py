import torch
from torch import nn
from typing import Optional

from .toto import Transformer
from .time_moe import TimeMoeInputEmbedding, TimeMoeRMSNorm
from jaxtyping import Float, Bool


class TSFormer(nn.Module):
    def __init__(
        self,
        input_channels: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
        causal: bool = False,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads."
        assert (d_model // n_heads) % 2 == 0, "RoPE requires even head_dim."

        self.input_channels = input_channels
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers

        self.embedding = TimeMoeInputEmbedding(
            input_size=1,
            hidden_size=d_model,
            hidden_act='silu'
        )
        self.mask_token = nn.Parameter(torch.zeros(d_model))

        self.transformer = Transformer(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=self.num_layers,
            mlp_hidden_dim=d_ff,
            dropout=dropout,
            spacewise_every_n_layers=2,
            spacewise_first=False,
            use_memory_efficient_attention=False,
            # fusion=self.fusion,
        )

        self.norm = TimeMoeRMSNorm(d_model)
        self.reconstruction_head = nn.Linear(d_model, 1)
        
    def forward(
        self,
        inputs: Float[torch.Tensor, "batch variate time_steps"],
        input_padding_mask: Bool[torch.Tensor, "batch variate time_steps"],
        id_mask: Float[torch.Tensor, "batch #variate time_steps"],
        # scaling_prefix_length: Optional[int] = None,
        # num_exogenous_variables: int = 0,
        point_mask: Optional[Bool[torch.Tensor, "batch variate time_steps"]] = None,
    ):
        # embedding
        embeddings: Float[torch.Tensor, "batch variate seq_len d_model"] = self.embedding(inputs.unsqueeze(-1))
        if point_mask is not None:
            if tuple(point_mask.shape) != tuple(inputs.shape):
                raise ValueError("point_mask must match inputs shape [batch, variate, time_steps]")
            mask_token = self.mask_token.to(dtype=embeddings.dtype, device=embeddings.device).view(1, 1, 1, -1)
            embeddings = torch.where(point_mask.bool().unsqueeze(-1), mask_token.expand_as(embeddings), embeddings)
        
        # transformer
        transformerd = self.transformer(embeddings)
        
        # head
        recon = self.reconstruction_head(transformerd)
        recon = recon.squeeze(-1)
        
        return recon
