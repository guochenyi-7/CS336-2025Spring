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

PROMPT_TEMPLATE = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
"""


def load_val_data(val_file_path: str) -> tuple[list[str], list[str]]:
    prompts = []
    ground_truths = []

    with open(val_file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            prompts.append(PROMPT_TEMPLATE.format(question=item["question"]))

            answer = item["answer"]
            if "####" in answer:
                answer = answer.split("####")[-1].strip()
            ground_truths.append(answer)

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
    model_id = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    validation_file = project_root / "data" / "gsm8k" / "test.jsonl"
    prompts, ground_truths = load_val_data(str(validation_file))
    output_file = project_root / "cs336_alignment" / "results.jsonl"

    llm = init_vllm(model_id, gpu_memory_utilization=0.8)
    accuracy = evaluate_vllm(
        llm,
        r1_zero_reward_fn,
        prompts,
        sampling_params,
        ground_truths,
        str(output_file),
    )
    print(f"Validation Accuracy: {accuracy:.2%}")
