from __future__ import annotations

import torch
from jaxtyping import Bool, Float
from torch import nn

from .embedding import PatchEmbedding
from .time_moe import TimeMoeRMSNorm
from .toto import Transformer


class TSFormer(nn.Module):
    def __init__(
        self,
        patch_size: int,
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
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")

        self.patch_size = int(patch_size)
        self.input_channels = int(input_channels)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.num_layers = int(num_layers)

        self.embedding = PatchEmbedding(patch_size=self.patch_size, embed_dim=self.d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.d_model))

        self.transformer = Transformer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=self.num_layers,
            mlp_hidden_dim=d_ff,
            dropout=dropout,
            spacewise_every_n_layers=2,
            spacewise_first=False,
            use_memory_efficient_attention=False,
        )

        self.norm = TimeMoeRMSNorm(self.d_model)
        self.reconstruction_head = nn.Linear(self.d_model, self.patch_size)

    def patchify(
        self, inputs: Float[torch.Tensor, "batch variate time_steps"]
    ) -> Float[torch.Tensor, "batch variate seq_len patch_size"]:
        if inputs.ndim != 3:
            raise ValueError(f"inputs must have shape [batch, variate, time_steps], got {tuple(inputs.shape)}")
        if inputs.shape[-1] % self.patch_size != 0:
            raise ValueError(
                f"input time length ({inputs.shape[-1]}) must be divisible by patch_size ({self.patch_size})"
            )
        return inputs.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)

    def unpatchify(
        self, patches: Float[torch.Tensor, "batch variate seq_len patch_size"]
    ) -> Float[torch.Tensor, "batch variate time_steps"]:
        if patches.ndim != 4:
            raise ValueError(f"patches must have shape [batch, variate, seq_len, patch_size], got {tuple(patches.shape)}")
        if patches.shape[-1] != self.patch_size:
            raise ValueError(f"patches last dim must equal patch_size={self.patch_size}, got {patches.shape[-1]}")
        batch_size, variate, seq_len, _ = patches.shape
        return patches.contiguous().view(batch_size, variate, seq_len * self.patch_size)

    def reconstruct_patches(
        self,
        inputs: Float[torch.Tensor, "batch variate time_steps"],
        input_padding_mask: Bool[torch.Tensor, "batch variate time_steps"],
        id_mask: torch.Tensor,
        patch_mask: Bool[torch.Tensor, "batch variate seq_len"] | None = None,
    ) -> Float[torch.Tensor, "batch variate seq_len patch_size"]:
        if tuple(input_padding_mask.shape) != tuple(inputs.shape):
            raise ValueError("input_padding_mask must match inputs shape [batch, variate, time_steps]")
        if tuple(id_mask.shape) != tuple(inputs.shape):
            raise ValueError("id_mask must match inputs shape [batch, variate, time_steps]")
        if not input_padding_mask.bool().all():
            raise ValueError("model_v3 expects padding to be applied before the backbone call")

        embeddings, patched_id_mask = self.embedding(inputs, id_mask.long())
        if patch_mask is not None:
            if tuple(patch_mask.shape) != tuple(embeddings.shape[:3]):
                raise ValueError("patch_mask must match patch token shape [batch, variate, seq_len]")
            mask_token = self.mask_token.to(dtype=embeddings.dtype, device=embeddings.device)
            embeddings = torch.where(patch_mask.bool().unsqueeze(-1), mask_token.expand_as(embeddings), embeddings)

        transformed = self.transformer(embeddings, id_mask=patched_id_mask)
        transformed = self.norm(transformed)
        return self.reconstruction_head(transformed)

    def forward(
        self,
        inputs: Float[torch.Tensor, "batch variate time_steps"],
        input_padding_mask: Bool[torch.Tensor, "batch variate time_steps"],
        id_mask: torch.Tensor,
        patch_mask: Bool[torch.Tensor, "batch variate seq_len"] | None = None,
    ) -> Float[torch.Tensor, "batch variate time_steps"]:
        recon_patches = self.reconstruct_patches(
            inputs=inputs,
            input_padding_mask=input_padding_mask,
            id_mask=id_mask,
            patch_mask=patch_mask,
        )
        return self.unpatchify(recon_patches)
