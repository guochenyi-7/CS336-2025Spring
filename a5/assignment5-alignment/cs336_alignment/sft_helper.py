import json
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

project_root = Path(__file__).parent.parent

DEFAULT_TRAIN_DATA_PATH = project_root / "data" / "MATH" / "sft.jsonl"
DEFAULT_FILTERED_TRAIN_DATA_PATH = project_root / "data" / "MATH" / "sft_filtered.jsonl"
DEFAULT_INPUT_PATH = DEFAULT_TRAIN_DATA_PATH
DEFAULT_OUTPUT_PATH = DEFAULT_FILTERED_TRAIN_DATA_PATH
DEFAULT_STATS_PATH = project_root / "data" / "MATH" / "sft_filtered_stats.json"

MODEL_ID = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
VALIDATION_FILE = project_root / "data" / "MATH" / "validation.jsonl"
DEFAULT_OUTPUT_FILE = project_root / "outputs" / "math_eval" / "baseline_results.jsonl"
PROMPT_PATH = project_root / "cs336_alignment" / "prompts" / "r1_zero.prompt"

with open(PROMPT_PATH, "r", encoding="utf-8") as file:
    PROMPT_TEMPLATE = file.read()


class SFTDataset(Dataset):
    def __init__(self, data_path: str | Path, num_samples: int | None = None):
        self.data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
                if num_samples is not None and len(self.data) >= num_samples:
                    break

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.data[idx]


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1)


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    masked_tensor = mask.to(tensor.dtype) * tensor
    if dim is None:
        summed = masked_tensor.sum()
    else:
        summed = masked_tensor.sum(dim=dim)
    return summed / normalize_constant


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    batch_full_ids = []
    batch_masks = []
    max_len = 0

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        output_ids = tokenizer(output, add_special_tokens=False)["input_ids"]

        full_ids = prompt_ids + output_ids
        max_len = max(max_len, len(full_ids))

        mask = [0] * len(prompt_ids) + [1] * len(output_ids)
        batch_full_ids.append(full_ids)
        batch_masks.append(mask)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    padded_ids = []
    padded_masks = []

    for ids, mask in zip(batch_full_ids, batch_masks):
        pad_len = max_len - len(ids)
        padded_ids.append(ids + [pad_token_id] * pad_len)
        padded_masks.append(mask + [0] * pad_len)

    full_tensor = torch.tensor(padded_ids, dtype=torch.long)
    mask_tensor = torch.tensor(padded_masks, dtype=torch.long)

    return {
        "input_ids": full_tensor[:, :-1],
        "labels": full_tensor[:, 1:],
        "response_mask": mask_tensor[:, 1:],
    }


def create_collate_fn(tokenizer: PreTrainedTokenizer):
    def collate_fn(batch: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        prompt_strs = [item["prompt"] for item in batch]
        output_strs = [item["response"] for item in batch]
        return tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)

    return collate_fn


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    log_probs_all = F.log_softmax(logits, dim=-1)
    labels_expanded = labels.unsqueeze(-1)
    log_probs = torch.gather(log_probs_all, dim=-1, index=labels_expanded).squeeze(-1)

    result = {
        "log_probs": log_probs,
    }
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    nll = -policy_log_probs
    per_example_loss = masked_normalize(
        tensor=nll,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=1,
    )
    loss = per_example_loss.mean()
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()

    return scaled_loss, {"unscaled_loss": loss.detach()}


def load_policy_model(
    model_id: str,
    device: str,
    attn_implementation: str = "flash_attention_2",
) -> tuple[AutoTokenizer, PreTrainedModel, str]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": attn_implementation,
    }
    try:
        policy_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).to(device)
        used_attn_implementation = attn_implementation
    except Exception as exc:
        if attn_implementation != "flash_attention_2":
            raise
        print(
            "flash_attention_2 unavailable, falling back to sdpa for policy model loading:",
            exc,
        )
        policy_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device)
        used_attn_implementation = "sdpa"

    policy_model.gradient_checkpointing_enable()
    policy_model.config.use_cache = False
    return tokenizer, policy_model, used_attn_implementation


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: Any) -> None:
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def resolve_train_data_path(use_filtered_data: bool) -> Path:
    return DEFAULT_FILTERED_TRAIN_DATA_PATH if use_filtered_data else DEFAULT_TRAIN_DATA_PATH


def format_value_for_name(value: float) -> str:
    return f"{value:.0e}".replace("+0", "").replace("-0", "-")


def build_run_name(
    num_train_samples: int | None,
    learning_rate: float,
    train_batch_size: int,
    use_filtered_data: bool,
) -> str:
    sample_tag = "full" if num_train_samples is None else str(num_train_samples)
    data_tag = "filtered" if use_filtered_data else "raw"
    lr_tag = format_value_for_name(learning_rate)
    return f"math_sft_{data_tag}_{sample_tag}_lr{lr_tag}_bs{train_batch_size}"


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
):
    from vllm import LLM
    from vllm.model_executor.utils import set_random_seed as vllm_set_random_seed

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


def build_eval_sampling_params():
    from vllm import SamplingParams

    return SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )


def evaluate_vllm(
    vllm_model: Any,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    eval_sampling_params: Any,
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


def evaluate_policy(
    policy_model: PreTrainedModel,
    vllm_engine: Any,
    val_prompts: list[str],
    val_ground_truths: list[str],
    eval_sampling_params: Any,
    eval_step: int,
) -> dict[str, object]:
    load_policy_into_vllm_instance(policy_model, vllm_engine)
    outputs = vllm_engine.generate(val_prompts, eval_sampling_params)
    generated_responses = [output.outputs[0].text for output in outputs]

    format_rewards: list[float] = []
    answer_rewards: list[float] = []
    total_rewards: list[float] = []
    for response, ground_truth in zip(generated_responses, val_ground_truths):
        scores = r1_zero_reward_fn(response, ground_truth)
        format_rewards.append(scores["format_reward"])
        answer_rewards.append(scores["answer_reward"])
        total_rewards.append(scores["reward"])

    return {
        "eval_step": eval_step,
        "eval/accuracy": sum(answer_rewards) / len(answer_rewards),
        "eval/mean_format_reward": sum(format_rewards) / len(format_rewards),
        "eval/mean_answer_reward": sum(answer_rewards) / len(answer_rewards),
        "eval/mean_total_reward": sum(total_rewards) / len(total_rewards),
    }


def filter_math_sft_dataset(
    input_file: str | Path,
    output_file: str | Path,
    stats_file: str | Path | None = None,
) -> dict[str, int]:
    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_examples = 0
    kept_examples = 0
    format_failures = 0
    wrong_answers = 0

    with input_file.open("r", encoding="utf-8") as in_f, output_file.open(
        "w",
        encoding="utf-8",
    ) as out_f:
        for line in in_f:
            total_examples += 1
            example = json.loads(line)
            ground_truth = example.get("ground_truth")
            if ground_truth is None:
                raise ValueError(
                    "Expected each SFT example to include a `ground_truth` field for filtering."
                )

            scores = r1_zero_reward_fn(example["response"], ground_truth)
            if scores["answer_reward"] == 1.0:
                out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
                kept_examples += 1
            elif scores["format_reward"] == 0.0:
                format_failures += 1
            else:
                wrong_answers += 1

    stats = {
        "input_examples": total_examples,
        "kept_examples": kept_examples,
        "filtered_examples": total_examples - kept_examples,
        "format_failures": format_failures,
        "wrong_answers": wrong_answers,
    }

    if stats_file is not None:
        stats_file = Path(stats_file)
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        with stats_file.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


def log_generations(
    prompts: list[str],
    responses: list[str],
    ground_truths: list[str],
    reward_fn,
    response_entropies: list[float],
    response_lengths: list[int],
    prefix: str = "eval",
    output_path: str | Path | None = None,
) -> dict[str, object]:
    import numpy as np
    import wandb

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
        f"{prefix}/mean_response_entropy": (
            float(np.mean(response_entropies)) if response_entropies else 0.0
        ),
        f"{prefix}/mean_response_length": float(np.mean(response_lengths)) if response_lengths else 0.0,
        f"{prefix}/mean_correct_response_length": (
            float(np.mean(correct_lengths)) if correct_lengths else 0.0
        ),
        f"{prefix}/mean_incorrect_response_length": (
            float(np.mean(incorrect_lengths)) if incorrect_lengths else 0.0
        ),
        f"{prefix}/generations": table,
    }


__all__ = [
    "compute_entropy",
    "build_eval_sampling_params",
    "DEFAULT_FILTERED_TRAIN_DATA_PATH",
    "DEFAULT_INPUT_PATH",
    "DEFAULT_OUTPUT_FILE",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_STATS_PATH",
    "DEFAULT_TRAIN_DATA_PATH",
    "MODEL_ID",
    "PROMPT_PATH",
    "PROMPT_TEMPLATE",
    "SFTDataset",
    "VALIDATION_FILE",
    "build_run_name",
    "create_collate_fn",
    "evaluate_vllm",
    "evaluate_policy",
    "extract_ground_truth",
    "filter_math_sft_dataset",
    "format_value_for_name",
    "get_response_log_probs",
    "init_vllm",
    "load_val_data",
    "load_policy_into_vllm_instance",
    "load_policy_model",
    "log_generations",
    "masked_normalize",
    "resolve_train_data_path",
    "sft_microbatch_train_step",
    "tokenize_prompt_and_output",
]
