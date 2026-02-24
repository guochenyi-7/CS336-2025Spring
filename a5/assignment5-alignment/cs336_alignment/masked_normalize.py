import torch

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    masked_tensor = mask.to(tensor.dtype) * tensor

    if dim is None:
        sumed = masked_tensor.sum()
    else:
        sumed = masked_tensor.sum(dim=dim)
    
    normalized_tensor = sumed / normalize_constant

    return normalized_tensor
