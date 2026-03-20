import argparse
import json
import sys
from pathlib import Path
import torch
from torch.nn.utils import clip_grad_norm_
import random
import wandb
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from typing import Literal

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cs336_alignment.drgrpo_grader import (
    question_only_reward_fn,
    r1_zero_reward_fn,
)
from cs336_alignment.grpo_helper import (
    compute_group_normalized_rewards,
    masked_mean,
    grpo_microbatch_train_step,
)

from cs336_alignment.sft_helper import (
    MODEL_ID,
    VALIDATION_FILE,
    init_vllm,
    load_policy_model,
    get_response_log_probs,
    load_policy_into_vllm_instance,
    tokenize_prompt_and_output,
    evaluate_vllm,
)

LossType = Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"]

TRAIN_FILE = project_root / "data" / "MATH" / "train.jsonl"
DEFAULT_OUTPUT_ROOT = project_root / "outputs" / "grpo"
PROMPT_FILES = {
    "r1_zero": project_root / "cs336_alignment" / "prompts" / "r1_zero.prompt",
    "question_only": project_root / "cs336_alignment" / "prompts" / "question_only.prompt",
}

@dataclass
class GRPOConfig:
    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6

    rollout_batch_size: int = 256
    group_size: int = 8

    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4
    sampling_max_tokens: int = 1024

    epochs_per_rollout_batch: int = 1
    train_batch_size: int = 256
    gradient_accumulation_steps: int = 128

    cliprange: float = 0.2
    max_grad_norm: float = 1.0

    loss_type: LossType = "reinforce_with_baseline"
    use_std_normalization: bool = True

    eval_every: int = 10
    num_eval_examples: int | None = None
    seed: int = 42
    device: str = "cuda"

def format_value_for_name(value: float) -> str:
    return f"{value:.0e}".replace("+0", "").replace("-0", "-")

def load_prompt_template(prompt_style: str) -> str:
    prompt_path = PROMPT_FILES[prompt_style]
    with prompt_path.open("r", encoding="utf-8") as f:
        return f.read()

def load_math_examples(
    data_path: str | Path,
    prompt_template: str,
    num_samples: int | None = None,
) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    data_path = Path(data_path)

    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            examples.append(
                {
                    "prompt": prompt_template.format(question=item["problem"]),
                    "ground_truth": item["answer"],
                }
            )
            if num_samples is not None and len(examples) >= num_samples:
                break

    return examples

def build_grpo_run_name(
    experiment: str,
    cfg: GRPOConfig,
    prompt_style: str,
    reward_style: str,
    num_train_samples: int | None,
) -> str:
    sample_tag = "full" if num_train_samples is None else str(num_train_samples)
    lr_tag = format_value_for_name(cfg.learning_rate)
    std_tag = "std" if cfg.use_std_normalization else "nostd"
    return (
        f"{experiment}_"
        f"{prompt_style}_{reward_style}_"
        f"{cfg.loss_type}_"
        f"{std_tag}_"
        f"s{sample_tag}_"
        f"lr{lr_tag}_"
        f"rb{cfg.rollout_batch_size}_"
        f"tb{cfg.train_batch_size}_"
        f"ep{cfg.epochs_per_rollout_batch}"
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRPO experiments on the MATH dataset.")
    parser.add_argument("--experiment", type=str, default="grpo", help="WandB group / experiment tag.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional explicit WandB run name.")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="cs336-assignment5-grpo",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=TRAIN_FILE,
        help="Path to the GRPO train jsonl file.",
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=VALIDATION_FILE,
        help="Path to the validation jsonl file.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=sorted(PROMPT_FILES),
        default="r1_zero",
        help="Prompt template to use for both training and evaluation.",
    )
    parser.add_argument(
        "--reward-style",
        choices=["auto", "r1_zero", "question_only"],
        default="auto",
        help="Reward function to use. 'auto' matches the prompt style.",
    )
    parser.add_argument(
        "--num-train-samples",
        type=int,
        default=None,
        help="Optional cap on the number of train examples to load.",
    )
    parser.add_argument("--n-grpo-steps", type=int, default=GRPOConfig.n_grpo_steps)
    parser.add_argument("--learning-rate", type=float, default=GRPOConfig.learning_rate)
    parser.add_argument("--advantage-eps", type=float, default=GRPOConfig.advantage_eps)
    parser.add_argument("--rollout-batch-size", type=int, default=GRPOConfig.rollout_batch_size)
    parser.add_argument("--group-size", type=int, default=GRPOConfig.group_size)
    parser.add_argument("--sampling-temperature", type=float, default=GRPOConfig.sampling_temperature)
    parser.add_argument("--sampling-min-tokens", type=int, default=GRPOConfig.sampling_min_tokens)
    parser.add_argument("--sampling-max-tokens", type=int, default=GRPOConfig.sampling_max_tokens)
    parser.add_argument(
        "--epochs-per-rollout-batch",
        type=int,
        default=GRPOConfig.epochs_per_rollout_batch,
    )
    parser.add_argument("--train-batch-size", type=int, default=GRPOConfig.train_batch_size)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=GRPOConfig.gradient_accumulation_steps,
    )
    parser.add_argument("--cliprange", type=float, default=GRPOConfig.cliprange)
    parser.add_argument("--max-grad-norm", type=float, default=GRPOConfig.max_grad_norm)
    parser.add_argument(
        "--loss-type",
        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        default=GRPOConfig.loss_type,
    )
    parser.add_argument(
        "--use-std-normalization",
        action=argparse.BooleanOptionalAction,
        default=GRPOConfig.use_std_normalization,
        help="Whether to divide group-centered rewards by the group std.",
    )
    parser.add_argument("--eval-every", type=int, default=GRPOConfig.eval_every)
    parser.add_argument(
        "--num-eval-examples",
        type=int,
        default=GRPOConfig.num_eval_examples,
        help="Optional cap on validation examples. Defaults to the full validation set.",
    )
    parser.add_argument("--seed", type=int, default=GRPOConfig.seed)
    parser.add_argument("--policy-device", type=str, default="cuda:0")
    parser.add_argument("--vllm-device", type=str, default="cuda:1")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Target fraction of GPU memory reserved by vLLM.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for optional model checkpoints.",
    )
    parser.add_argument(
        "--save-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to save the final policy checkpoint.",
    )
    return parser.parse_args()

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sample_prompt_batch(train_examples, n_prompts: int):
    batch = random.sample(train_examples, n_prompts)
    prompts = [ex["prompt"] for ex in batch]
    ground_truths = [ex["ground_truth"] for ex in batch]
    return prompts, ground_truths

def flatten_vllm_outputs(outputs, group_size: int) -> list[str]:
    """
    Assume llm.generate(..., SamplingParams(n=group_size)) returns one RequestOutput per prompt,
    and each RequestOutput has .outputs = list[CompletionOutput] of length group_size.
    """
    responses = []
    for req_out in outputs:
        assert len(req_out.outputs) == group_size
        for out in req_out.outputs:
            responses.append(out.text)
    return responses

def repeat_each(xs, n):
    out = []
    for x in xs:
        out.extend([x] * n)
    return out

def build_rollout_dataset(
    prompts: list[str],
    responses: list[str],
    ground_truths: list[str],
    raw_rewards: torch.Tensor,         # (rollout_batch_size,)
    advantages: torch.Tensor,          # (rollout_batch_size,)
):
    repeated_prompts = repeat_each(prompts, len(responses) // len(prompts))
    repeated_ground_truths = repeat_each(ground_truths, len(responses) // len(prompts))

    return {
        "prompt_strs": repeated_prompts,
        "output_strs": responses,
        "ground_truths": repeated_ground_truths,
        "raw_rewards": raw_rewards,     # shape (N,)
        "advantages": advantages,       # shape (N,)
    }

def prepare_rollout_batch_tensors(
    rollout_data: dict,
    tokenizer: PreTrainedTokenizerBase,
    policy: PreTrainedModel,
    cfg: GRPOConfig,
    need_old_log_probs: bool,
):
    tokenized = tokenize_prompt_and_output(
        rollout_data["prompt_strs"],
        rollout_data["output_strs"],
        tokenizer,
    )

    input_ids = tokenized["input_ids"].to(cfg.device)
    labels = tokenized["labels"].to(cfg.device)
    response_mask = tokenized["response_mask"].to(cfg.device)

    with torch.inference_mode():
        old_log_probs = None
        if need_old_log_probs:
            old_out = get_response_log_probs(
                model=policy,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=False,
            )
            old_log_probs = old_out["log_probs"]

    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
        "raw_rewards": rollout_data["raw_rewards"].to(cfg.device).unsqueeze(1),   # (N, 1)
        "advantages": rollout_data["advantages"].to(cfg.device).unsqueeze(1),     # (N, 1)
    }

    if old_log_probs is not None:
        batch["old_log_probs"] = old_log_probs

    return batch

def iterate_minibatches(tensor_dict: dict[str, torch.Tensor], batch_size: int):
    n = next(iter(tensor_dict.values())).shape[0]
    indices = torch.randperm(n, device=next(iter(tensor_dict.values())).device)

    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        yield {k: v[idx] for k, v in tensor_dict.items()}


def iterate_microbatches(tensor_dict: dict[str, torch.Tensor], microbatch_size: int):
    n = next(iter(tensor_dict.values())).shape[0]
    for start in range(0, n, microbatch_size):
        yield {k: v[start:start + microbatch_size] for k, v in tensor_dict.items()}

def log_dict(prefix: str, step: int, metrics: dict):
    logged = {f"{prefix}/{k}": float(v) if torch.is_tensor(v) else v for k, v in metrics.items()}
    logged[f"{prefix}_step"] = step
    wandb.log(logged)

def grpo_train_loop(
    policy: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    vllm_model: LLM,
    train_examples: list[dict],
    val_examples: list[dict],
    reward_fn,
    optimizer: torch.optim.Optimizer,
    cfg: GRPOConfig,
):
    set_seed(cfg.seed)
    policy.train()

    assert cfg.train_batch_size % cfg.gradient_accumulation_steps == 0
    micro_train_batch_size = cfg.train_batch_size // cfg.gradient_accumulation_steps
 
    assert cfg.rollout_batch_size % cfg.group_size == 0
    n_prompts_per_rollout_batch = cfg.rollout_batch_size // cfg.group_size

    assert cfg.train_batch_size >= cfg.group_size

    sampling_params = SamplingParams(
        temperature=cfg.sampling_temperature,
        top_p=1.0,
        min_tokens=cfg.sampling_min_tokens,
        max_tokens=cfg.sampling_max_tokens,
        n=cfg.group_size,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=cfg.seed,
    )

    wandb_started_here = False
    if wandb.run is None:
        wandb.init(
            project="cs336-assignment5-grpo",
            config=cfg.__dict__,
        )
        wandb_started_here = True

    wandb.define_metric("rollout_step")
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("rollout/*", step_metric="rollout_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    global_update_step = 0

    for grpo_step in range(cfg.n_grpo_steps):
        # ---------------------------------------------------------
        # 1) Sample prompts
        # ---------------------------------------------------------
        prompts, ground_truths = sample_prompt_batch(
            train_examples,
            n_prompts=n_prompts_per_rollout_batch,
        )

        # ---------------------------------------------------------
        # 2) Sync current policy -> vLLM and rollout
        # ---------------------------------------------------------
        policy.eval()
        load_policy_into_vllm_instance(policy, vllm_model)

        rollout_outputs = vllm_model.generate(prompts, sampling_params)
        rollout_responses = flatten_vllm_outputs(rollout_outputs, cfg.group_size)

        repeated_ground_truths = repeat_each(ground_truths, cfg.group_size)

        # ---------------------------------------------------------
        # 3) Compute rewards / advantages
        # ---------------------------------------------------------
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=cfg.group_size,
            advantage_eps=cfg.advantage_eps,
            normalize_by_std=cfg.use_std_normalization,
        )

        rollout_log = {
            "raw_reward_mean": raw_rewards.mean().item(),
            "raw_reward_std": raw_rewards.std().item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
        }
        rollout_log.update(reward_metadata)

        log_dict("rollout", grpo_step, rollout_log)

        rollout_data = build_rollout_dataset(
            prompts=prompts,
            responses=rollout_responses,
            ground_truths=ground_truths,
            raw_rewards=raw_rewards,
            advantages=advantages,
        )

        # ---------------------------------------------------------
        # 4) Tokenize prompt+response and cache old_log_probs if needed
        # ---------------------------------------------------------
        need_old_log_probs = (cfg.loss_type == "grpo_clip")
        train_tensors = prepare_rollout_batch_tensors(
            rollout_data=rollout_data,
            tokenizer=tokenizer,
            policy=policy,
            cfg=cfg,
            need_old_log_probs=need_old_log_probs,
        )

        # ---------------------------------------------------------
        # 5) Inner training loop
        # ---------------------------------------------------------
        policy.train()

        for epoch in range(cfg.epochs_per_rollout_batch):
            for minibatch in iterate_minibatches(train_tensors, cfg.train_batch_size):
                optimizer.zero_grad(set_to_none=True)

                microbatch_logs = []

                for microbatch in iterate_microbatches(minibatch, micro_train_batch_size):
                    out = get_response_log_probs(
                        model=policy,
                        input_ids=microbatch["input_ids"],
                        labels=microbatch["labels"],
                        return_token_entropy=True,
                    )
                    policy_log_probs = out["log_probs"]
                    token_entropy = out["token_entropy"]

                    loss, metadata = grpo_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=microbatch["response_mask"],
                        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                        loss_type=cfg.loss_type,
                        raw_rewards=microbatch.get("raw_rewards"),
                        advantages=microbatch.get("advantages"),
                        old_log_probs=microbatch.get("old_log_probs"),
                        cliprange=cfg.cliprange if cfg.loss_type == "grpo_clip" else None,
                    )
                    with torch.no_grad():
                        avg_entropy = masked_mean(
                            token_entropy,
                            microbatch["response_mask"],
                            dim=None,
                        )
                    mb_log = {
                        "loss": metadata["unscaled_loss"].detach(),
                        "scaled_loss": loss.detach(),
                        "token_entropy": avg_entropy.detach(),
                    }
                    for k, v in metadata.items():
                        if k == "unscaled_loss":
                            continue
                        if torch.is_tensor(v):
                            if v.numel() == 1:
                                mb_log[k] = v.detach()
                            elif k == "is_clipped":
                                mb_log["clip_frac"] = v.float().mean().detach()
                        else:
                            mb_log[k] = v
                    microbatch_logs.append(mb_log)

                grad_norm = clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()
                global_update_step += 1

                # Aggregate logs
                train_log = {
                    "grad_norm": grad_norm.detach() if torch.is_tensor(grad_norm) else grad_norm,
                    "raw_reward_mean": minibatch["raw_rewards"].mean().item(),
                    "advantage_mean": minibatch["advantages"].mean().item(),
                    "advantage_std": minibatch["advantages"].std(unbiased=False).item(),
                }

                # average microbatch logs
                if microbatch_logs:
                    keys = microbatch_logs[0].keys()
                    for k in keys:
                        vals = [x[k] for x in microbatch_logs]
                        vals = [
                            v.item() if torch.is_tensor(v) and v.numel() == 1 else v
                            for v in vals
                        ]
                        if isinstance(vals[0], (int, float)):
                            train_log[k] = sum(vals) / len(vals)

                log_dict("train", global_update_step, train_log)

        # ---------------------------------------------------------
        # 6) Periodic evaluation
        # ---------------------------------------------------------
        if (grpo_step + 1) % cfg.eval_every == 0:
            policy.eval()
            load_policy_into_vllm_instance(policy, vllm_model)
            eval_examples = (
                val_examples
                if cfg.num_eval_examples is None
                else val_examples[:cfg.num_eval_examples]
            )

            eval_metrics = evaluate_vllm(
                vllm_model=vllm_model,
                reward_fn=reward_fn,
                prompts=[ex["prompt"] for ex in eval_examples],
                eval_sampling_params=SamplingParams(
                    temperature=1.0,
                    top_p=1.0,
                    max_tokens=cfg.sampling_max_tokens,
                    min_tokens=cfg.sampling_min_tokens,
                    stop=["</answer>"],
                    include_stop_str_in_output=True,
                    seed=cfg.seed,
                ),
                ground_truths=[ex["ground_truth"] for ex in eval_examples],
                output_file=None,
            )

            log_dict("eval", grpo_step + 1, eval_metrics)
            print(
                f"GRPO step {grpo_step + 1} | "
                f"Validation Accuracy: {eval_metrics['accuracy']:.2%}"
            )

    if wandb_started_here:
        wandb.finish()

    return policy

if __name__ == "__main__":
    args = parse_args()

    resolved_reward_style = (
        args.prompt_style if args.reward_style == "auto" else args.reward_style
    )
    reward_fn = (
        r1_zero_reward_fn
        if resolved_reward_style == "r1_zero"
        else question_only_reward_fn
    )

    cfg = GRPOConfig(
        n_grpo_steps=args.n_grpo_steps,
        learning_rate=args.learning_rate,
        advantage_eps=args.advantage_eps,
        rollout_batch_size=args.rollout_batch_size,
        group_size=args.group_size,
        sampling_temperature=args.sampling_temperature,
        sampling_min_tokens=args.sampling_min_tokens,
        sampling_max_tokens=args.sampling_max_tokens,
        epochs_per_rollout_batch=args.epochs_per_rollout_batch,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        cliprange=args.cliprange,
        max_grad_norm=args.max_grad_norm,
        loss_type=args.loss_type,
        use_std_normalization=args.use_std_normalization,
        eval_every=args.eval_every,
        num_eval_examples=args.num_eval_examples,
        seed=args.seed,
        device=args.policy_device,
    )

    prompt_template = load_prompt_template(args.prompt_style)
    train_examples = load_math_examples(
        args.train_file,
        prompt_template,
        num_samples=args.num_train_samples,
    )
    val_examples = load_math_examples(args.validation_file, prompt_template)

    run_name = args.run_name or build_grpo_run_name(
        experiment=args.experiment,
        cfg=cfg,
        prompt_style=args.prompt_style,
        reward_style=resolved_reward_style,
        num_train_samples=args.num_train_samples,
    )
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Starting GRPO run '{run_name}' with {len(train_examples)} train examples "
        f"and {len(val_examples)} validation examples."
    )

    tokenizer, policy, _ = load_policy_model(
        MODEL_ID,
        device=args.policy_device,
    )
    vllm_model = init_vllm(
        MODEL_ID,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        device=args.vllm_device,
    )
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    wandb.init(
        project=args.wandb_project,
        group=args.experiment,
        name=run_name,
        config={
            **cfg.__dict__,
            "experiment": args.experiment,
            "prompt_style": args.prompt_style,
            "reward_style": resolved_reward_style,
            "train_file": str(args.train_file),
            "validation_file": str(args.validation_file),
            "num_train_samples": args.num_train_samples,
            "policy_device": args.policy_device,
            "vllm_device": args.vllm_device,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "save_model": args.save_model,
        },
    )

    try:
        trained_policy = grpo_train_loop(
            policy=policy,
            tokenizer=tokenizer,
            vllm_model=vllm_model,
            train_examples=train_examples,
            val_examples=val_examples,
            reward_fn=reward_fn,
            optimizer=optimizer,
            cfg=cfg,
        )

        if args.save_model:
            save_dir = output_dir / "final_model"
            save_dir.mkdir(parents=True, exist_ok=True)
            trained_policy.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"Saved final model to {save_dir}")
    finally:
        if wandb.run is not None:
            wandb.finish()
