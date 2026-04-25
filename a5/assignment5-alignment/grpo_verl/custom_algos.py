from __future__ import annotations

from typing import Any, Optional

import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss, register_adv_est, register_policy_loss
from verl.workers.config import ActorConfig

ASSIGNMENT_REINFORCE_LOSS = "assignment_reinforce"
ASSIGNMENT_GRPO_CLIP_LOSS = "assignment_grpo_clip"
ASSIGNMENT_GRPO_ADV = "assignment_grpo"
ASSIGNMENT_RAW_REWARD_ADV = "assignment_raw_reward"


def _global_batch_info(config: Optional[ActorConfig]) -> dict[str, Any]:
    if config is None:
        return {}
    return getattr(config, "global_batch_info", {}) or {}


@register_policy_loss(ASSIGNMENT_REINFORCE_LOSS)
def compute_assignment_reinforce_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    del rollout_is_weights

    pg_losses = -advantages * log_prob
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **_global_batch_info(config),
    )

    approx_kl = verl_F.masked_mean(old_log_prob - log_prob, response_mask)
    zero = torch.tensor(0.0, device=pg_loss.device)
    metrics = {
        "actor/pg_clipfrac": zero.detach().item(),
        "actor/pg_clipfrac_lower": zero.detach().item(),
        "actor/ppo_kl": approx_kl.detach().item(),
    }
    return pg_loss, metrics


@register_policy_loss(ASSIGNMENT_GRPO_CLIP_LOSS)
def compute_assignment_grpo_clip_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    del rollout_is_weights

    if config is None:
        raise ValueError("Actor config is required for assignment GRPO-clip loss.")

    clip_ratio = config.clip_ratio
    ratio = torch.exp(log_prob - old_log_prob)
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

    pg_losses_unclipped = -advantages * ratio
    pg_losses_clipped = -advantages * clipped_ratio
    pg_losses = torch.maximum(pg_losses_unclipped, pg_losses_clipped)

    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **_global_batch_info(config),
    )

    approx_kl = verl_F.masked_mean(old_log_prob - log_prob, response_mask)
    clipfrac = verl_F.masked_mean(torch.gt(pg_losses_clipped, pg_losses_unclipped).float(), response_mask)
    metrics = {
        "actor/pg_clipfrac": clipfrac.detach().item(),
        "actor/pg_clipfrac_lower": torch.tensor(0.0, device=pg_loss.device).detach().item(),
        "actor/ppo_kl": approx_kl.detach().item(),
    }
    return pg_loss, metrics


@register_adv_est(ASSIGNMENT_GRPO_ADV)
def compute_assignment_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index,
    config=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    del kwargs

    if config is None:
        raise ValueError("Algorithm config is required for assignment GRPO advantage.")

    epsilon = config.get("advantage_eps", 1e-6)
    normalize_by_std = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)

    grouped_scores: dict[Any, list[torch.Tensor]] = {}
    group_means: dict[Any, torch.Tensor] = {}
    group_stds: dict[Any, torch.Tensor] = {}

    with torch.no_grad():
        for i, group_id in enumerate(index):
            grouped_scores.setdefault(group_id, []).append(scores[i])

        for group_id, reward_list in grouped_scores.items():
            reward_tensor = torch.stack(reward_list)
            if reward_tensor.numel() == 1:
                group_means[group_id] = torch.tensor(0.0, device=reward_tensor.device)
                group_stds[group_id] = torch.tensor(1.0, device=reward_tensor.device)
            else:
                group_means[group_id] = reward_tensor.mean()
                group_stds[group_id] = reward_tensor.std()

        normalized_scores = torch.empty_like(scores)
        for i, group_id in enumerate(index):
            centered = scores[i] - group_means[group_id]
            if normalize_by_std:
                normalized_scores[i] = centered / (group_stds[group_id] + epsilon)
            else:
                normalized_scores[i] = centered

        advantages = normalized_scores.unsqueeze(-1) * response_mask

    return advantages, advantages


@register_adv_est(ASSIGNMENT_RAW_REWARD_ADV)
def compute_assignment_raw_reward_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    del kwargs

    scores = token_level_rewards.sum(dim=-1, keepdim=True)
    repeated_scores = scores * response_mask
    return repeated_scores, repeated_scores
