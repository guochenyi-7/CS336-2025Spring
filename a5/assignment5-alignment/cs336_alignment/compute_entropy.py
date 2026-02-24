import torch
import torch.nn.functional as F

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)

    entropy = -torch.sum(probs * log_probs, dim=-1)

    return entropy
