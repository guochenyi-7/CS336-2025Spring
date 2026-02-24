import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from cs336_alignment.compute_entropy import compute_entropy

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    
    logits = model(input_ids).logits
    log_probs_all = F.log_softmax(logits, dim=-1)
    labels_expended = labels.unsqueeze(-1)
    log_probs = torch.gather(log_probs_all, dim=-1, index=labels_expended).squeeze(-1)

    result = {
        "log_probs": log_probs,
    }

    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)

    return result
    