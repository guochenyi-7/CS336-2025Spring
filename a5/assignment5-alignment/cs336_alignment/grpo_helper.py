import torch
from typing import Callable

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
