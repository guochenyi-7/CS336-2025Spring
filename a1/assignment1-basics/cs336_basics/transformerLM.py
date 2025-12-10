import torch
import torch.nn as nn

from torch import Tensor
from jaxtyping import Float, Int

from .rmsnorm import RMSNorm
from .attention import MultiheadSelfAttentionWithRoPE
from .swiglu import SwiGLUFFN
from .embedding import Embedding
from .linear import Linear


class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int, 
            max_seq_len: int,
            theta: float,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
        ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads,
        self.d_ff = d_ff

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttentionWithRoPE(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.ffn = SwiGLUFFN(d_model, d_ff, device, dtype)

    def forward(
            self, 
            x: Float[Tensor, "batch seq_len d_model"],
            token_positions: Int[Tensor, "... seq_len"] | None = None,
        ) -> Float[Tensor, "batch seq_len d_model"]:
        if token_positions is None:
            token_positions = torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
            self, 
            vocab_size: int,
            context_size: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
    
        para_dtype = (
            dtype
            if (
                dtype is not None
                and torch.is_floating_point(torch.tensor([], dtype=dtype))
            ) 
            else torch.float32
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_size,
                theta=rope_theta,
                device=device,
                dtype=para_dtype ,
            ) for _ in range(num_layers)
        ])

        self.token_embeddings = Embedding(vocab_size, d_model, device, para_dtype)
        self.ln_final = RMSNorm(d_model, device=device, dtype=para_dtype)
        self.lm_head = Linear(d_model, vocab_size, device, para_dtype)
    
    def forward(
            self,
            input_indices: Int[Tensor, "batch seq_len"],
            token_positions: Int[Tensor, "batch seq_len"] | None = None,
        ) -> Float[Tensor, "batch seq_len vocab_size"]:
        if token_positions is None:
            token_positions = torch.arange(input_indices.shape[1], device=input_indices.device).expand(input_indices.shape[0], -1)

        x = self.token_embeddings(input_indices)

        for layer in self.layers:
            x = layer(x, token_positions)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
