import torch
from jaxtyping import Float
from torch import Tensor

def softmax(x: Float[Tensor, "..."], dim: int=-1) -> Float[Tensor, "..."]:
    if x.device.type == "mps":
        compute_dtype = torch.float32
    else:
        compute_dtype = torch.float64
    in_type = x.dtype
    x = x.to(compute_dtype)
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    x_exp = torch.exp(x_shifted)
    x_sum = x_exp.sum(dim=dim, keepdim=True)
    x_out = x_exp / x_sum
    return x_out.to(in_type)