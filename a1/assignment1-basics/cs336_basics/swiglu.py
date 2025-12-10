import torch
import torch.nn as nn

from einops import einsum
from jaxtyping import Float
from torch import Tensor

from .linear import Linear

def SiLU(x):
    in_type = x.dtype
    x = x.to(torch.float32)
    x = x * torch.sigmoid(x)
    return x.to(in_type)

class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,    
        ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1_weight = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.w2_weight = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        self.w3_weight = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        std = (2.0 / (d_ff + d_model)) ** 0.5
        nn.init.trunc_normal_(self.w1_weight, mean=0, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.w2_weight, mean=0, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.w3_weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(
            self,
            x: Float[Tensor, " ... d_model"],
        ) -> Float[Tensor, " ... d_model"]:
        in_type = x.dtype
        x = x.to(torch.float32)
        w1_x = einsum(x, self.w1_weight, "... d_model, d_ff d_model -> ... d_ff")
        w3_x = einsum(x, self.w3_weight, "... d_model, d_ff d_model -> ... d_ff")

        out = SiLU(w1_x) * w3_x
        out = einsum(out, self.w2_weight, "... d_ff, d_model d_ff -> ... d_model")
        return out.to(in_type) #type: ignore
    
class SwiGLUFFN(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,    
        ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
    
    def forward(
            self,
            x: Float[Tensor, " ... d_model"],
        ) -> Float[Tensor, " ... d_model"]:
        gate = SiLU(self.w1(x)) * self.w3(x)
        return self.w2(gate)
