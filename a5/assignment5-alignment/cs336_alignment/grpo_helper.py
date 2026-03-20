import torch
from typing import Callable, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    raw_rewards = []
    for response, ground_truth in zip(rollout_responses,repeated_ground_truths):
        scores = reward_fn(response, ground_truth)
        raw_rewards.append(scores["reward"])

    raw_rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float32)
    if raw_rewards_tensor.numel() % group_size != 0:
        raise ValueError("Number of rewards must be divisible by group_size")
    
    grouped_rewards = raw_rewards_tensor.view(-1, group_size)
    group_means = grouped_rewards.mean(dim=1, keepdim=True)
    

    if normalize_by_std:
        grouped_stds = grouped_rewards.std(dim=1, keepdim=True)
        grouped_advantages = (grouped_rewards - group_means) / (grouped_stds + advantage_eps)
    else:
        grouped_advantages = grouped_rewards - group_means

    advantages = grouped_advantages.reshape(-1)
    metadata = {
        "mean_reward": raw_rewards_tensor.mean().item(),
        "std_reward": raw_rewards_tensor.std().item(),
        "max_reward": raw_rewards_tensor.max().item(),
        "min_reward": raw_rewards_tensor.min().item(),
    }

    return advantages, raw_rewards_tensor, metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict [str, torch.Tensor]]:
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)

    unclipped_obj = ratio * advantages
    clipped_obj = clipped_ratio * advantages

    objective = torch.minimum(unclipped_obj, clipped_obj)
    loss = -objective

    metadata = {
        "is_clipped": clipped_obj < unclipped_obj,
    }

    return loss, metadata

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(
            raw_rewards,
            policy_log_probs,
        )
        metadata = {}
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(
            advantages,
            policy_log_probs,
        )
        metadata = {}
    elif loss_type == "grpo_clip":
        loss, metadata = compute_grpo_clip_loss(
            advantages,
            policy_log_probs,
            old_log_probs,
            cliprange,
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    return loss, metadata

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    mask = mask.to(tensor.dtype)
    masked_tensor = mask * tensor
    if dim is None:
        summed = masked_tensor.sum()
        count = mask.sum()
    else:
        summed = masked_tensor.sum(dim=dim)
        count = mask.sum(dim=dim)
    return summed / count

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    per_example_loss = masked_mean(
        per_token_loss,
        response_mask,
        dim=1,
    )

    loss = per_example_loss.mean()
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()

    metadata = dict(metadata)
    metadata["unscaled_loss"] = loss.detach()
    metadata["per_example_loss"] = per_example_loss.detach().mean()
    
    return scaled_loss.detach(), metadata
