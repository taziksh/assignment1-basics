import torch.nn as nn
from jaxtyping import Bool, Float, Int
import torch
from einops import einsum, rearrange
import math

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        stdev = math.sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=stdev, a=-3*stdev, b=3*stdev)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
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
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones((d_model), device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt((x**2).sum(dim=-1, keepdim=True)/self.d_model + self.eps)
        result = (x * self.weight) / rms
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
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: Float[torch.Tensor, " ... d_model"]):
        return self.w2(SiLU(self.w1(x)) * self.w3(x))
    
class RoPE(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device: torch.device | None = None
    ) -> None:
        super().__init__()
        i = torch.arange(max_seq_len, device=device).unsqueeze(-1)
        k = torch.arange(1, d_k/2+1, device=device).unsqueeze(0)
        Theta = i / (theta**((2*k-2)/d_k))
        self.register_buffer("cos", torch.cos(Theta), persistent=False)
        self.register_buffer("sin", torch.sin(Theta), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_pos = self.cos[token_positions]
        sin_pos = self.sin[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_even_new = x_even * cos_pos - x_odd * sin_pos
        x_odd_new = x_even * sin_pos + x_odd * cos_pos
        return torch.stack([x_even_new, x_odd_new], dim=-1).flatten(-2, -1)
    
def softmax(in_features: Float[torch.Tensor, " ..."], dim: int) -> Float[torch.Tensor, " ..."]:
    c = torch.max(in_features, dim=dim, keepdim=True).values
    return torch.exp(in_features - c) / torch.sum(torch.exp(in_features - c), dim=dim, keepdim=True)

def scaled_dot_product_attention(
        Q: Float[torch.Tensor, " ... queries d_k"],
        K: Float[torch.Tensor, " ... keys d_k"],
        V: Float[torch.Tensor, " ... keys d_v"],
        mask: Bool[torch.Tensor, " ... queries keys"] | None = None
) -> Float[torch.Tensor, " ... queries d_v"]:
    
    d_k = Q.shape[-1]
    scores = einsum(Q, K, " ... queries d_k, ... keys d_k -> ... queries keys")
    scores = scores/math.sqrt(d_k)

    if mask is not None:
        scores.masked_fill_(~mask, -float('inf'))

    sm = softmax(in_features=scores, dim=-1)
    return einsum(sm, V, "... queries keys, ... keys d_v -> ... queries d_v")

class MHASelfAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            rope: RoPE | None = None
    ) -> None:
        super().__init__()
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model//num_heads
        self.rope = rope
    
    def forward(self, 
            input_features: Float[torch.Tensor, " ... s d_model"],
            token_positions: Int[torch.Tensor, " ... s"] | None = None
        ) -> Float[torch.Tensor, " ... s d_model"]:
        Q = rearrange(self.q_proj(input_features), " ... s (h d_k) -> ... h s d_k", h=self.num_heads, d_k=self.d_k)
        K = rearrange(self.k_proj(input_features), " ... s (h d_k) -> ... h s d_k", h=self.num_heads, d_k=self.d_k)
        V = rearrange(self.v_proj(input_features), " ... s (h d_k) -> ... h s d_k", h=self.num_heads, d_k=self.d_k)
        s = input_features.shape[-2]
        mask = torch.tril(torch.ones(s, s, device=input_features.device, dtype=bool))        

        if self.rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        attn = scaled_dot_product_attention(Q, K, V, mask)
        attn = rearrange(attn, " ... h s d_k -> ... s (h d_k)")
        return self.output_proj(attn)

class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int,
            theta: float
    ) -> None:
        super().__init__()
        
        self.ln1 = RMSNorm(d_model)

        d_k = d_model//num_heads
        rope = RoPE(theta, d_k, max_seq_len)
        self.attn = MHASelfAttention(d_model, num_heads, rope)
        
        self.ln2 = RMSNorm(d_model)

        self.ffn = SwiGLU(d_model, d_ff)


    def forward(self, x):
        s = x.shape[-2]
        token_positions = torch.arange(s)
        y = x + self.attn(input_features=self.ln1(x), token_positions=token_positions)
        return y + self.ffn(self.ln2(y))
    
class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float
    ) -> None:
        super().__init__()

        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            block = TransformerBlock(d_model, num_heads, d_ff, max_seq_len=context_length, theta=rope_theta)
            self.layers.append(block)
        
        self.ln_final = RMSNorm(d_model)

        self.lm_head = Linear(d_model, vocab_size)  

    def forward(self, in_indices: Int[torch.Tensor, " b s"]) -> Float[torch.Tensor, " b s vocab_size"]:
        out = self.token_embeddings(in_indices)
        
        for layer in self.layers:
            out = layer(out)
        
        out = self.ln_final(out)

        out = self.lm_head(out)

        return out