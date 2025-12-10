import torch
import torch.nn as nn
import math

from torch.optim import Optimizer
from typing import Optional, Callable, Iterable

class MyAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.9) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None): # type: ignore
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["step"] += 1
                t = state["step"]
                m = state["m"]
                v = state["v"]

                m.mul_(beta1).add_(grad, alpha=1-beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                bias_corection1 = 1 - beta1 ** t
                bias_corection2 = 1 - beta2 ** t
                step_size = lr * math.sqrt(bias_corection2) / bias_corection1

                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-step_size)

                if wd != 0:
                    p.data.add_(p.data, alpha=-lr*wd)

        return loss


def lr_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
):
    if it < warmup_iters:
        cur_lr = it / warmup_iters * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        delta = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi
        cur_lr = min_learning_rate + 0.5 * (1 + math.cos(delta)) * (max_learning_rate - min_learning_rate)
    else:
        cur_lr = min_learning_rate
    return cur_lr


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    params = [p for p in parameters if p.grad is not None]

    if len(params) == 0:
        return
    
    sum_squares = sum(p.grad.detach().pow(2).sum() for p in params) # type: ignore
    total_norm = sum_squares.sqrt() # type: ignore

    if total_norm > max_l2_norm:
        epsilon = 1e-6
        clip = max_l2_norm / (total_norm + epsilon)
        for p in params:
            p.grad.mul_(clip) # type: ignore
