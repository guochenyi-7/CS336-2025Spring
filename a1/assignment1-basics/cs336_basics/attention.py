import torch
import torch.nn as nn
from einops import einsum, rearrange
from torch import Tensor
from jaxtyping import Float

from .softmax import softmax
from .linear import Linear
from .rope import RoPE

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
            self,
            Q: Float[Tensor, "... queries d_k"],
            K: Float[Tensor, "... keys d_k"],
            V: Float[Tensor, "... values d_v"], 
            mask: Float[Tensor, "... queries d_k"] | None = None, 
        ) -> Float[Tensor, "... queries d_v"]:
        d_k = Q.shape[-1]
        scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / (d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask==False, float("-inf"))

        attn_scores = softmax(scores, dim=-1)
        out = einsum(attn_scores, V, "... queries keys, ... keys d_v -> ... queries d_v")
        return out


class MultiheadSelfAttention(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            num_heads: int,
        ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj  = Linear(d_model, d_model)
        self.v_proj  = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()
    
    def forward(
            self, 
            x: Float[Tensor, "batch seq_len d_model"],
        ) -> Float[Tensor, "batch seq_len d_model"]:
        seq_len = x.shape[-2]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = rearrange(Q, "batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim", num_heads=self.num_heads)
        K = rearrange(K, "batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim", num_heads=self.num_heads)
        V = rearrange(V, "batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim", num_heads=self.num_heads)

        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))
        attn = self.attention(Q, K, V, mask)

        attn = rearrange(attn, "batch num_heads seq_len head_dim -> batch seq_len (num_heads head_dim)", num_heads=self.num_heads)

        return self.output_proj(attn)


class MultiheadSelfAttentionWithRoPE(nn.Module):
    causal_mask: torch.Tensor
    
    def __init__(
            self, 
            d_model: int, 
            num_heads: int,
            theta: float,
            max_seq_len: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()
        self.RoPE = RoPE(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len, device=device, dtype=dtype)

        # 预先生成一个足够大的下三角 Mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        # 注册为buffer，这样它会被视为模型状态的一部分，不需要每次计算
        self.register_buffer("causal_mask", mask)

    def forward(
            self, 
            x: Float[Tensor, "batch seq_len d_model"],
            token_positions: torch.Tensor,
        ) -> Float[Tensor, "batch seq_len d_model"]:
        seq_len = x.shape[-2]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = rearrange(Q, "batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim", num_heads=self.num_heads)
        K = rearrange(K, "batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim", num_heads=self.num_heads)
        V = rearrange(V, "batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim", num_heads=self.num_heads)
        Q = self.RoPE(Q, token_positions)
        K = self.RoPE(K, token_positions)

        # mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))
        mask = self.causal_mask[:seq_len, :seq_len]
        attn = self.attention(Q, K, V, mask)

        attn = rearrange(attn, "batch num_heads seq_len head_dim -> batch seq_len (num_heads head_dim)", num_heads=self.num_heads)

        return self.output_proj(attn)
