import torch
from einops import rearrange

def make_batched_block_mask(t: torch.Tensor) -> torch.Tensor:
    unsqueezed = rearrange(t, "... d -> ... 1 d")
    return unsqueezed == unsqueezed.transpose(-1, -2)
