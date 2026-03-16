import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "8"

import torch
import wandb
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.evaluate_llm import (
    MODEL_ID,
    VALIDATION_FILE,
    build_eval_sampling_params,
    init_vllm,
    load_val_data,
)
from cs336_alignment.get_response_log_probs import get_response_log_probs
from cs336_alignment.sft_microbatch_train_step import sft_microbatch_train_step
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output

DEFAULT_TRAIN_DATA_PATH = project_root / "data" / "MATH" / "sft.jsonl"
DEFAULT_FILTERED_TRAIN_DATA_PATH = project_root / "data" / "MATH" / "sft_filtered.jsonl"
DEFAULT_OUTPUT_ROOT = project_root / "outputs" / "sft"


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


def create_collate_fn(tokenizer):
    def collate_fn(batch):
        prompt_strs = [item["prompt"] for item in batch]
        output_strs = [item["response"] for item in batch]
        return tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)

    return collate_fn


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


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm) -> None:
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


def evaluate_policy(
    policy_model: PreTrainedModel,
    vllm_engine,
    val_prompts: list[str],
    val_ground_truths: list[str],
    eval_sampling_params,
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

    metrics: dict[str, object] = {
        "eval_step": eval_step,
        "eval/accuracy": sum(answer_rewards) / len(answer_rewards),
        "eval/mean_format_reward": sum(format_rewards) / len(format_rewards),
        "eval/mean_answer_reward": sum(answer_rewards) / len(answer_rewards),
        "eval/mean_total_reward": sum(total_rewards) / len(total_rewards),
    }

    return metrics


def run_sft_experiment(
    num_train_samples: int | None = None,
    learning_rate: float = 1e-5,
    train_batch_size: int = 8,
    use_filtered_data: bool = False,
) -> Path:
    seed = 42
    gradient_accumulation_steps = 2
    num_epochs = 10
    eval_interval = 50
    train_device = "cuda:0"
    vllm_device = "cuda:1"
    gpu_memory_utilization = 0.85
    wandb_project = "cs336-assignment5-sft"
    train_data_path = resolve_train_data_path(use_filtered_data)
    wandb_run_name = build_run_name(
        num_train_samples=num_train_samples,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        use_filtered_data=use_filtered_data,
    )
    output_dir = DEFAULT_OUTPUT_ROOT / wandb_run_name

    output_dir.mkdir(parents=True, exist_ok=True)
    train_data_path = Path(train_data_path)

    torch.manual_seed(seed)
    random.seed(seed)

    tokenizer, policy_model, _ = load_policy_model(
        MODEL_ID,
        device=train_device,
    )
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
    eval_sampling_params = build_eval_sampling_params()
    vllm_engine = init_vllm(
        MODEL_ID,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
        device=vllm_device,
    )

    train_dataset = SFTDataset(train_data_path, num_samples=num_train_samples)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=create_collate_fn(tokenizer),
    )
    val_prompts, val_ground_truths = load_val_data(VALIDATION_FILE)

    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    policy_model.train()

    for epoch in range(num_epochs):
        microbatches_in_group = 0
        current_group_size = gradient_accumulation_steps
        for batch_idx, batch in enumerate(train_dataloader):
            if microbatches_in_group == 0:
                current_group_size = min(
                    gradient_accumulation_steps,
                    len(train_dataloader) - batch_idx,
                )

            input_ids = batch["input_ids"].to(train_device)
            labels = batch["labels"].to(train_device)
            response_mask = batch["response_mask"].to(train_device)

            log_prob_dict = get_response_log_probs(
                model=policy_model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=True,
            )
            scaled_loss, metadata = sft_microbatch_train_step(
                policy_log_probs=log_prob_dict["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=current_group_size,
            )
            microbatches_in_group += 1

            response_mask_float = response_mask.float()
            response_lengths = response_mask_float.sum(dim=1).clamp_min(1.0)
            mean_response_entropy = (
                (log_prob_dict["token_entropy"] * response_mask_float).sum(dim=1)
                / response_lengths
            ).mean()

            if microbatches_in_group == current_group_size:
                clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                train_metrics = {
                    "train_step": global_step,
                    "train/loss": metadata["unscaled_loss"].item(),
                    "train/mean_response_entropy": mean_response_entropy.item(),
                    "train/mean_response_length": response_lengths.mean().item(),
                    "train/epoch": epoch + 1,
                }
                wandb.log(train_metrics)

                if global_step % eval_interval == 0:
                    policy_model.eval()
                    eval_metrics = evaluate_policy(
                        policy_model=policy_model,
                        vllm_engine=vllm_engine,
                        val_prompts=val_prompts,
                        val_ground_truths=val_ground_truths,
                        eval_sampling_params=eval_sampling_params,
                        eval_step=global_step,
                    )
                    wandb.log(eval_metrics)
                    print(
                        f"Step {global_step} | Validation Accuracy: {eval_metrics['eval/accuracy']:.2%}"
                    )
                    policy_model.train()

                microbatches_in_group = 0

    save_dir = output_dir / "final_model"
    save_dir.mkdir(parents=True, exist_ok=True)
    policy_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    wandb.finish()
    return save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-train-samples",
        type=int,
        default=None,
        help="Optional cap on the number of unique SFT examples to load.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate to use for SFT.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=8,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--use-filtered-data",
        action="store_true",
        help="Use data/MATH/sft_filtered.jsonl instead of the raw MATH SFT data.",
    )
    args = parser.parse_args()

    run_sft_experiment(
        num_train_samples=args.num_train_samples,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        use_filtered_data=args.use_filtered_data,
    )
 
