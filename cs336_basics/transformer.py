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

class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.W, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device: torch.device | None = None,
            dtype: torch.device | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(torch.ones((d_model), device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt((x**2).sum(dim=-1, keepdim=True)/self.d_model + self.eps)
        result = (x * self.g) / rms
        return result.to(in_dtype)
    
def SiLU(in_features: Float[torch.Tensor, " ..."]) -> Float[torch.Tensor, " ..."]:
    return in_features * torch.sigmoid(in_features)

class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            device: torch.device | None = None,
            dtype: torch.device | None = None
    ) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))

    def forward(self, x: Float[torch.Tensor, " ... d_model"]):
        return einsum(
            self.w2, 
            SiLU(einsum(
                self.w1, 
                x, 
                "d_ff d_model, ... d_model -> d_ff ..."
            )) * 
            einsum(
                self.w3, 
                x, 
                "d_ff d_model, ... d_model -> d_ff ..."
            ),
            "d_model d_ff, d_ff ... -> ... d_model"
        )