import json
from pathlib import Path
from typing import Callable

import numpy as np
import wandb


def log_generations(
    prompts: list[str],
    responses: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    response_entropies: list[float],
    response_lengths: list[int],
    prefix: str = "eval",
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Summarize a set of generations for wandb logging and optional JSONL export."""
    if not (
        len(prompts)
        == len(responses)
        == len(ground_truths)
        == len(response_entropies)
        == len(response_lengths)
    ):
        raise ValueError("All generation logging inputs must have the same length.")

    format_rewards: list[float] = []
    answer_rewards: list[float] = []
    total_rewards: list[float] = []
    correct_lengths: list[int] = []
    incorrect_lengths: list[int] = []
    rows: list[list[object]] = []
    records: list[dict[str, object]] = []

    for prompt, response, ground_truth, entropy, length in zip(
        prompts,
        responses,
        ground_truths,
        response_entropies,
        response_lengths,
    ):
        scores = reward_fn(response, ground_truth)
        format_reward = scores.get("format_reward", 0.0)
        answer_reward = scores.get("answer_reward", 0.0)
        total_reward = scores.get("reward", 0.0)

        format_rewards.append(format_reward)
        answer_rewards.append(answer_reward)
        total_rewards.append(total_reward)
        if answer_reward == 1.0:
            correct_lengths.append(length)
        else:
            incorrect_lengths.append(length)

        rows.append(
            [
                prompt,
                response,
                ground_truth,
                format_reward,
                answer_reward,
                total_reward,
                entropy,
                length,
            ]
        )
        records.append(
            {
                "prompt": prompt,
                "generated_text": response,
                "ground_truth": ground_truth,
                "scores": scores,
                "mean_response_entropy": entropy,
                "response_length": length,
            }
        )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    table = wandb.Table(
        columns=[
            "Prompt",
            "Response",
            "Ground Truth",
            "Format Reward",
            "Answer Reward",
            "Total Reward",
            "Mean Response Entropy",
            "Response Length",
        ],
        data=rows,
    )

    return {
        f"{prefix}/mean_format_reward": float(np.mean(format_rewards)) if format_rewards else 0.0,
        f"{prefix}/mean_answer_reward": float(np.mean(answer_rewards)) if answer_rewards else 0.0,
        f"{prefix}/mean_total_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
        f"{prefix}/mean_response_entropy": float(np.mean(response_entropies)) if response_entropies else 0.0,
        f"{prefix}/mean_response_length": float(np.mean(response_lengths)) if response_lengths else 0.0,
        f"{prefix}/mean_correct_response_length": (
            float(np.mean(correct_lengths)) if correct_lengths else 0.0
        ),
        f"{prefix}/mean_incorrect_response_length": (
            float(np.mean(incorrect_lengths)) if incorrect_lengths else 0.0
        ),
        f"{prefix}/generations": table,
    }
    
