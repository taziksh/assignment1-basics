import torch.nn as nn
from jaxtyping import Bool, Float
import torch
from einops import einsum
import math

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        stdev = math.sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(self.W, mean=0, std=stdev, a=-3*stdev, b=3*stdev)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")
        return y