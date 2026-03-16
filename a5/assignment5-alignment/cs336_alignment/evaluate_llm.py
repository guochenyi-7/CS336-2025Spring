import argparse
import json
import sys
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import torch
from vllm import LLM, SamplingParams
from vllm.model_executor.utils import set_random_seed as vllm_set_random_seed

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

MODEL_ID = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
VALIDATION_FILE = project_root / "data" / "MATH" / "validation.jsonl"
DEFAULT_OUTPUT_FILE = project_root / "outputs" / "math_eval" / "baseline_results.jsonl"
PROMPT_PATH = project_root / "cs336_alignment" / "prompts" / "r1_zero.prompt"

with open(PROMPT_PATH, "r", encoding="utf-8") as file:
    PROMPT_TEMPLATE = file.read()


def extract_ground_truth(answer: str) -> str:
    answer = answer.strip()
    if "####" in answer:
        answer = answer.split("####")[-1].strip()
    return answer


def load_val_data(val_file_path: str | Path) -> tuple[list[str], list[str]]:
    prompts: list[str] = []
    ground_truths: list[str] = []

    with open(val_file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            prompts.append(PROMPT_TEMPLATE.format(question=item["problem"]))
            ground_truths.append(extract_ground_truth(item["answer"]))

    return prompts, ground_truths


def init_vllm(
    model_id: str,
    gpu_memory_utilization: float = 0.85,
    seed: int = 42,
    device: str | None = None,
) -> LLM:
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    llm_kwargs = {
        "model": model_id,
        "dtype": torch.bfloat16,
        "enable_prefix_caching": True,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    if device is not None:
        llm_kwargs["device"] = device
    with world_size_patch, profiling_patch:
        return LLM(**llm_kwargs)


def build_eval_sampling_params() -> SamplingParams:
    return SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    eval_sampling_params: SamplingParams,
    ground_truths: list[str],
    output_file: str | Path | None,
) -> dict[str, float]:
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    format_rewards: list[float] = []
    answer_rewards: list[float] = []
    total_rewards: list[float] = []

    for prompt, ground_truth, output_obj in zip(prompts, ground_truths, outputs):
        generated_text = output_obj.outputs[0].text
        scores = reward_fn(generated_text, ground_truth)

        format_rewards.append(scores["format_reward"])
        answer_rewards.append(scores["answer_reward"])
        total_rewards.append(scores["reward"])
        results.append(
            {
                "prompt": prompt,
                "ground_truth": ground_truth,
                "generated_text": generated_text,
                "scores": scores,
            }
        )

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

    return {
        "accuracy": sum(answer_rewards) / len(answer_rewards),
        "mean_format_reward": sum(format_rewards) / len(format_rewards),
        "mean_answer_reward": sum(answer_rewards) / len(answer_rewards),
        "mean_total_reward": sum(total_rewards) / len(total_rewards),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Where to write per-example evaluation results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by vLLM.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM GPU memory utilization fraction.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional vLLM device, e.g. cuda:1.",
    )
    args = parser.parse_args()

    prompts, ground_truths = load_val_data(VALIDATION_FILE)
    llm = init_vllm(
        MODEL_ID,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        device=args.device,
    )
    metrics = evaluate_vllm(
        llm,
        r1_zero_reward_fn,
        prompts,
        build_eval_sampling_params(),
        ground_truths,
        args.output_file,
    )

    print(f"Validation Accuracy: {metrics['accuracy']:.2%}")
    print(f"Mean Format Reward: {metrics['mean_format_reward']:.4f}")
    print(f"Mean Total Reward: {metrics['mean_total_reward']:.4f}")
