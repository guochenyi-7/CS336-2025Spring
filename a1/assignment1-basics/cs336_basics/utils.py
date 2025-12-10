import torch
import os
import numpy as np
from torch import Tensor
from jaxtyping import Float, Int
from typing import IO, BinaryIO

def cross_entropy_loss(
        logits: Float[Tensor, "batch_size vocab_size"],
        target: Int[Tensor, "batch_size"],
    ) -> Float[Tensor, ""]:
    max_logits, _ = logits.max(dim=-1, keepdim=True)
    shifted_logits = logits - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))
    target_logits = shifted_logits.gather(dim=-1, index=target.unsqueeze(-1))
    target_logits = target_logits.squeeze(-1)
    ele_loss = log_sum_exp - target_logits
    loss = ele_loss.mean()
    return loss

def get_batch(
        dataset: np.ndarray,
        batch_size: int,
        context_length: int,
        device: str = "mps"
):
    high = len(dataset) - context_length
    starts = torch.randint(low=0, high=high, size=(batch_size,)).tolist()

    input = [dataset[i: i + context_length] for i in starts]
    target = [dataset[i + 1 : i + context_length + 1] for i in starts]

    x_stack = np.stack(input)
    y_stack = np.stack(target)

    x = torch.from_numpy(x_stack.astype(np.int64)).to(device)
    y = torch.from_numpy(y_stack.astype(np.int64)).to(device)

    return x, y

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | BinaryIO | IO[bytes]
):
    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint_data, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
):
    checkpoint_data = torch.load(src)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
    return int(checkpoint_data["iteration"])