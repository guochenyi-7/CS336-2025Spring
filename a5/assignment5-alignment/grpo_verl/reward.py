from __future__ import annotations

from collections.abc import Callable

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn

RewardFn = Callable[[str, str, bool], dict[str, float]]


REWARD_FNS: dict[str, RewardFn] = {
    "r1_zero": r1_zero_reward_fn,
    "question_only": question_only_reward_fn,
}


def _resolve_reward_fn(reward_style: str) -> RewardFn:
    if reward_style == "auto":
        reward_style = "r1_zero"

    if reward_style not in REWARD_FNS:
        raise ValueError(f"Unknown reward_style: {reward_style}. Expected one of {sorted(REWARD_FNS)}")

    return REWARD_FNS[reward_style]


def _zero_reward_dict() -> dict[str, float]:
    return {
        "score": 0.0,
        "format_reward": 0.0,
        "answer_reward": 0.0,
    }


def my_reward_fn(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    fast: bool = True,
    reward_style: str = "r1_zero",
):
    del data_source, extra_info

    if solution_str is None:
        return _zero_reward_dict()

    reward_fn = _resolve_reward_fn(reward_style)

    try:
        scores = reward_fn(
            response=str(solution_str),
            ground_truth=ground_truth,
            fast=fast,
        )
        return {
            "score": float(scores["reward"]),
            "format_reward": float(scores["format_reward"]),
            "answer_reward": float(scores["answer_reward"]),
        }
    except Exception:
        return _zero_reward_dict()


