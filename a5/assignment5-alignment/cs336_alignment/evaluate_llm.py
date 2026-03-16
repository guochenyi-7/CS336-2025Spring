import argparse
import sys
from pathlib import Path
from unittest.mock import patch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import torch
from vllm import LLM, SamplingParams
from vllm.model_executor.utils import set_random_seed as vllm_set_random_seed
from typing import Callable, List
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

PROMPT_PATH = project_root / "cs336_alignment" / "prompts" / "r1_zero.prompt"
with open(PROMPT_PATH, "r", encoding="utf-8") as file:
    PROMPT_TEMPLATE = file.read()

DATASET_CONFIGS = {
    "gsm8k": {
        "validation_file": project_root / "data" / "gsm8k" / "test.jsonl",
        "question_key": "question",
    },
    "math": {
        "validation_file": project_root / "data" / "MATH" / "validation.jsonl",
        "question_key": "problem",
    },
}


def get_dataset_config(dataset_name: str) -> dict:
    normalized_name = dataset_name.lower()
    if normalized_name not in DATASET_CONFIGS:
        supported = ", ".join(sorted(DATASET_CONFIGS))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected one of: {supported}.")
    return DATASET_CONFIGS[normalized_name]


def extract_ground_truth(answer: str) -> str:
    answer = answer.strip()
    if "####" in answer:
        answer = answer.split("####")[-1].strip()
    return answer


def load_val_data(val_file_path: str, question_key: str) -> tuple[list[str], list[str]]:
    prompts = []
    ground_truths = []

    with open(val_file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            prompts.append(PROMPT_TEMPLATE.format(question=item[question_key]))
            ground_truths.append(extract_ground_truth(item["answer"]))

    return prompts, ground_truths


def init_vllm(
    model_id: str,
    gpu_memory_utilization: float = 0.8,
    seed: int = 42,
) -> LLM:
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    ground_truths: List[str],
    output_file: str,
) -> float:
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    correct_count = 0

    for i, output_obj in enumerate(outputs):
        generated_text = output_obj.outputs[0].text
        ground_truth = ground_truths[i]

        scores = reward_fn(generated_text, ground_truth)
        if scores["answer_reward"] == 1.0:
            correct_count += 1

        results_entry = {
            "prompt": prompts[i],
            "ground_truth": ground_truth,
            "generated_text": generated_text,
            "scores": scores,
        }

        results.append(results_entry)

    with open(output_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    return correct_count / len(prompts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIGS),
        default="math",
        help="Dataset to evaluate on.",
    )
    args = parser.parse_args()
    dataset_config = get_dataset_config(args.dataset)

    model_id = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    validation_file = dataset_config["validation_file"]
    prompts, ground_truths = load_val_data(
        str(validation_file),
        question_key=dataset_config["question_key"],
    )
    output_file = project_root / "cs336_alignment" / "results.jsonl"

    llm = init_vllm(model_id, gpu_memory_utilization=0.85)
    accuracy = evaluate_vllm(
        llm,
        r1_zero_reward_fn,
        prompts,
        sampling_params,
        ground_truths,
        str(output_file),
    )
    print(f"Validation Accuracy: {accuracy:.2%}")
