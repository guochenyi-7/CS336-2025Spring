from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRPO_VERL_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(GRPO_VERL_DIR) not in sys.path:
    sys.path.insert(0, str(GRPO_VERL_DIR))


DEFAULT_TRAIN_FILE = GRPO_VERL_DIR / "data" / "train.parquet"
DEFAULT_VALIDATION_FILE = GRPO_VERL_DIR / "data" / "validation.parquet"
DEFAULT_CONFIG_FILE = GRPO_VERL_DIR / "config.yaml"
DEFAULT_MODEL_PATH = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
REWARD_FILE = GRPO_VERL_DIR / "reward.py"

ASSIGNMENT_REINFORCE_LOSS = "assignment_reinforce"
ASSIGNMENT_GRPO_CLIP_LOSS = "assignment_grpo_clip"
ASSIGNMENT_GRPO_ADV = "assignment_grpo"
ASSIGNMENT_RAW_REWARD_ADV = "assignment_raw_reward"


def format_value_for_name(value: float) -> str:
    return f"{value:.0e}".replace("+0", "").replace("-0", "-")


def build_run_name(
    experiment: str,
    loss_type: str,
    reward_style: str,
    num_train_samples: int | None,
    learning_rate: float,
    rollout_batch_size: int,
    train_batch_size: int,
    group_size: int,
    use_std_normalization: bool,
) -> str:
    sample_tag = "full" if num_train_samples is None or num_train_samples < 0 else str(num_train_samples)
    reward_tag = "r1_zero" if reward_style == "auto" else reward_style
    lr_tag = format_value_for_name(learning_rate)
    std_tag = "std" if use_std_normalization else "nostd"
    return (
        f"{experiment}_"
        f"{reward_tag}_"
        f"{loss_type}_"
        f"{std_tag}_"
        f"s{sample_tag}_"
        f"rb{rollout_batch_size}_"
        f"tb{train_batch_size}_"
        f"g{group_size}_"
        f"lr{lr_tag}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run handwritten-style GRPO with the verl trainer stack.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def _assert_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def _import_verl_stack() -> dict[str, Any]:
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        import custom_algos

        try:
            from verl.experimental.reward_loop import migrate_legacy_reward_impl
        except ImportError:
            migrate_legacy_reward_impl = lambda config: config

        from verl.trainer import main_ppo as verl_main_ppo
        from verl.trainer.main_ppo import run_ppo

        try:
            from verl.utils.device import auto_set_device
        except ImportError:
            auto_set_device = lambda config: config
    except ModuleNotFoundError as exc:
        missing = exc.name or "required package"
        raise ModuleNotFoundError(
            "Running grpo_verl/train.py requires a verl environment with "
            f"`verl`, `hydra-core`, and `omegaconf` installed. Missing module: {missing}."
        ) from exc

    return {
        "compose": compose,
        "initialize_config_dir": initialize_config_dir,
        "OmegaConf": OmegaConf,
        "migrate_legacy_reward_impl": migrate_legacy_reward_impl,
        "verl_main_ppo": verl_main_ppo,
        "run_ppo": run_ppo,
        "auto_set_device": auto_set_device,
    }


def _load_user_config(omega_conf_cls) -> Any:
    return omega_conf_cls.load(DEFAULT_CONFIG_FILE)


def _load_base_ppo_config(verl_stack: dict[str, Any]) -> Any:
    verl_config_dir = Path(verl_stack["verl_main_ppo"].__file__).resolve().parent / "config"
    with verl_stack["initialize_config_dir"](
        config_dir=str(verl_config_dir),
        version_base=None,
    ):
        return verl_stack["compose"](config_name="ppo_trainer")


def _resolve_optional_count(value: Any) -> int:
    if value is None:
        return -1
    value = int(value)
    return -1 if value < 0 else value


def _resolve_logger_list(user_cfg: Any, omega_conf_cls) -> list[str]:
    logger = omega_conf_cls.to_container(user_cfg.run.logger, resolve=True)
    if isinstance(logger, str):
        return [logger]
    if not isinstance(logger, list) or not logger:
        raise ValueError("run.logger must be a non-empty string or list of strings.")
    return [str(item) for item in logger]


def _resolve_batch_mapping(user_cfg: Any) -> tuple[int, int, int]:
    rollout_batch_size = int(user_cfg.grpo.rollout_batch_size)
    group_size = int(user_cfg.grpo.group_size)
    train_batch_size = int(user_cfg.grpo.train_batch_size)
    gradient_accumulation_steps = int(user_cfg.grpo.gradient_accumulation_steps)
    world_size = int(user_cfg.run.nnodes) * int(user_cfg.run.n_gpus_per_node)

    if rollout_batch_size % group_size != 0:
        raise ValueError("rollout_batch_size must be divisible by group_size.")
    if train_batch_size % group_size != 0:
        raise ValueError("train_batch_size must be divisible by group_size.")
    if train_batch_size % gradient_accumulation_steps != 0:
        raise ValueError("train_batch_size must be divisible by gradient_accumulation_steps.")
    if world_size <= 0:
        raise ValueError("nnodes * n_gpus_per_node must be positive.")

    prompts_per_rollout_batch = rollout_batch_size // group_size
    prompts_per_update_minibatch = train_batch_size // group_size
    actual_microbatch = train_batch_size // gradient_accumulation_steps

    if actual_microbatch % world_size != 0:
        raise ValueError(
            "Per-update microbatch size must be divisible by total actor GPUs. "
            f"Got actual_microbatch={actual_microbatch}, world_size={world_size}."
        )

    microbatch_per_gpu = actual_microbatch // world_size

    if prompts_per_update_minibatch > prompts_per_rollout_batch:
        raise ValueError(
            "train_batch_size cannot exceed rollout_batch_size once mapped into prompt groups. "
            f"Got train_batch_size={train_batch_size}, rollout_batch_size={rollout_batch_size}."
        )

    return prompts_per_rollout_batch, prompts_per_update_minibatch, microbatch_per_gpu


def _apply_dot_override_allowing_new_child(
    config: Any,
    key: str,
    value: Any,
    omega_conf_cls,
) -> None:
    # Some verl releases structure config nodes before declaring extension mappings.
    parent_key = key.rsplit(".", 1)[0]
    parent = config if parent_key == key else omega_conf_cls.select(config, parent_key)
    if parent is None:
        omega_conf_cls.update(config, key, value, merge=True)
        return

    previous_struct = omega_conf_cls.is_struct(parent)
    target = omega_conf_cls.select(config, key)
    previous_target_struct = None if target is None else omega_conf_cls.is_struct(target)
    omega_conf_cls.set_struct(parent, False)
    if target is not None:
        omega_conf_cls.set_struct(target, False)
    try:
        omega_conf_cls.update(config, key, value, merge=True)
    finally:
        if target is not None:
            omega_conf_cls.set_struct(target, previous_target_struct)
        omega_conf_cls.set_struct(parent, previous_struct)


def _apply_dot_overrides(
    config: Any,
    overrides: dict[str, Any],
    omega_conf_cls,
    allow_new_child_keys: set[str] | None = None,
) -> None:
    allow_new_child_keys = allow_new_child_keys or set()
    for key, value in overrides.items():
        if key in allow_new_child_keys:
            _apply_dot_override_allowing_new_child(config, key, value, omega_conf_cls)
        else:
            omega_conf_cls.update(config, key, value, merge=True)


def _apply_common_overrides(config: Any, user_cfg: Any, omega_conf_cls) -> None:
    prompt_batch_size, prompt_update_batch_size, microbatch_per_gpu = _resolve_batch_mapping(user_cfg)

    run_name = user_cfg.run.run_name or build_run_name(
        experiment=str(user_cfg.run.experiment),
        loss_type=str(user_cfg.grpo.loss_type),
        reward_style=str(user_cfg.grpo.reward_style),
        num_train_samples=(
            None
            if user_cfg.run.num_train_samples is None
            else int(user_cfg.run.num_train_samples)
        ),
        learning_rate=float(user_cfg.grpo.learning_rate),
        rollout_batch_size=int(user_cfg.grpo.rollout_batch_size),
        train_batch_size=int(user_cfg.grpo.train_batch_size),
        group_size=int(user_cfg.grpo.group_size),
        use_std_normalization=bool(user_cfg.grpo.use_std_normalization),
    )

    train_max_samples = _resolve_optional_count(user_cfg.run.num_train_samples)
    val_max_samples = _resolve_optional_count(user_cfg.run.num_eval_examples)
    test_freq = -1 if int(user_cfg.run.eval_every) <= 0 else int(user_cfg.run.eval_every)
    logger = _resolve_logger_list(user_cfg, omega_conf_cls)

    updates = {
        "data.train_files": str(DEFAULT_TRAIN_FILE),
        "data.val_files": str(DEFAULT_VALIDATION_FILE),
        "data.train_max_samples": train_max_samples,
        "data.val_max_samples": val_max_samples,
        "data.prompt_key": "prompt",
        "data.reward_fn_key": "data_source",
        "data.max_prompt_length": int(user_cfg.generation.max_prompt_length),
        "data.max_response_length": int(user_cfg.generation.max_response_length),
        "data.train_batch_size": prompt_batch_size,
        "data.val_batch_size": prompt_batch_size,
        "data.return_raw_input_ids": False,
        "data.return_raw_chat": True,
        "data.return_full_prompt": False,
        "data.shuffle": True,
        "data.validation_shuffle": False,
        "data.dataloader_num_workers": int(user_cfg.verl.dataloader_num_workers),
        "data.seed": int(user_cfg.run.seed),
        "data.filter_overlong_prompts": False,
        "data.truncation": "error",
        "data.trust_remote_code": bool(user_cfg.verl.trust_remote_code),
        "actor_rollout_ref.hybrid_engine": bool(user_cfg.verl.hybrid_engine),
        "actor_rollout_ref.model.path": DEFAULT_MODEL_PATH,
        "actor_rollout_ref.model.trust_remote_code": bool(user_cfg.verl.trust_remote_code),
        "actor_rollout_ref.model.enable_gradient_checkpointing": bool(
            user_cfg.verl.enable_gradient_checkpointing
        ),
        "actor_rollout_ref.model.use_remove_padding": bool(user_cfg.verl.use_remove_padding),
        "actor_rollout_ref.actor.strategy": str(user_cfg.verl.actor_strategy),
        "actor_rollout_ref.actor.ppo_mini_batch_size": prompt_update_batch_size,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": microbatch_per_gpu,
        "actor_rollout_ref.actor.ppo_epochs": int(user_cfg.grpo.epochs_per_rollout_batch),
        "actor_rollout_ref.actor.clip_ratio": float(user_cfg.grpo.cliprange),
        "actor_rollout_ref.actor.loss_agg_mode": str(user_cfg.verl.actor_loss_agg_mode),
        "actor_rollout_ref.actor.entropy_coeff": float(user_cfg.verl.actor_entropy_coeff),
        "actor_rollout_ref.actor.use_kl_loss": bool(user_cfg.grpo.use_kl_loss),
        "actor_rollout_ref.actor.use_torch_compile": bool(user_cfg.verl.use_torch_compile),
        "actor_rollout_ref.actor.shuffle": True,
        "actor_rollout_ref.actor.data_loader_seed": int(user_cfg.run.seed),
        "actor_rollout_ref.actor.optim.lr": float(user_cfg.grpo.learning_rate),
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": microbatch_per_gpu,
        "actor_rollout_ref.rollout.prompt_length": int(user_cfg.generation.max_prompt_length),
        "actor_rollout_ref.rollout.response_length": int(user_cfg.generation.max_response_length),
        "actor_rollout_ref.rollout.temperature": float(user_cfg.generation.temperature),
        "actor_rollout_ref.rollout.top_k": int(user_cfg.generation.top_k),
        "actor_rollout_ref.rollout.top_p": float(user_cfg.generation.top_p),
        "actor_rollout_ref.rollout.do_sample": bool(user_cfg.generation.do_sample),
        "actor_rollout_ref.rollout.n": int(user_cfg.grpo.group_size),
        "actor_rollout_ref.rollout.tensor_model_parallel_size": int(
            user_cfg.verl.tensor_model_parallel_size
        ),
        "actor_rollout_ref.rollout.gpu_memory_utilization": float(
            user_cfg.generation.gpu_memory_utilization
        ),
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": microbatch_per_gpu,
        "actor_rollout_ref.rollout.max_num_batched_tokens": int(user_cfg.verl.max_num_batched_tokens),
        "actor_rollout_ref.rollout.max_num_seqs": int(user_cfg.verl.max_num_seqs),
        "actor_rollout_ref.rollout.ignore_eos": bool(user_cfg.verl.ignore_eos),
        "actor_rollout_ref.rollout.calculate_log_probs": False,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": bool(user_cfg.generation.do_sample),
        "actor_rollout_ref.rollout.val_kwargs.temperature": float(user_cfg.generation.temperature),
        "actor_rollout_ref.rollout.val_kwargs.top_k": int(user_cfg.generation.top_k),
        "actor_rollout_ref.rollout.val_kwargs.top_p": float(user_cfg.generation.top_p),
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "reward.num_workers": int(user_cfg.verl.reward_num_workers),
        "reward.custom_reward_function.path": str(REWARD_FILE),
        "reward.custom_reward_function.name": "my_reward_fn",
        "reward.custom_reward_function.reward_kwargs": {
            "fast": bool(user_cfg.grpo.fast_reward),
            "reward_style": str(user_cfg.grpo.reward_style),
        },
        "reward.reward_manager.source": "register",
        "reward.reward_manager.name": "naive",
        "reward.reward_model.enable": False,
        "algorithm.norm_adv_by_std_in_grpo": bool(user_cfg.grpo.use_std_normalization),
        "algorithm.advantage_eps": float(user_cfg.grpo.advantage_eps),
        "algorithm.gamma": 1.0,
        "algorithm.lam": 1.0,
        "algorithm.use_kl_in_reward": False,
        "trainer.total_training_steps": int(user_cfg.grpo.n_grpo_steps),
        "trainer.total_epochs": max(1, int(user_cfg.grpo.n_grpo_steps)),
        "trainer.project_name": str(user_cfg.run.wandb_project),
        "trainer.experiment_name": run_name,
        "trainer.logger": logger,
        "trainer.log_val_generations": int(user_cfg.run.log_val_generations),
        "trainer.nnodes": int(user_cfg.run.nnodes),
        "trainer.n_gpus_per_node": int(user_cfg.run.n_gpus_per_node),
        "trainer.balance_batch": bool(user_cfg.run.balance_batch),
        "trainer.test_freq": test_freq,
        "trainer.save_freq": int(user_cfg.verl.save_freq),
        "trainer.critic_warmup": int(user_cfg.verl.critic_warmup),
        "trainer.val_before_train": bool(user_cfg.verl.val_before_train),
        "trainer.resume_mode": str(user_cfg.verl.resume_mode),
    }

    _apply_dot_overrides(
        config,
        updates,
        omega_conf_cls,
        allow_new_child_keys={"reward.custom_reward_function.reward_kwargs"},
    )

    omega_conf_cls.update(
        config,
        "actor_rollout_ref.actor.fsdp_config.seed",
        int(user_cfg.run.seed),
        merge=True,
    )
    omega_conf_cls.update(
        config,
        "actor_rollout_ref.ref.fsdp_config.seed",
        int(user_cfg.run.seed),
        merge=True,
    )

    loss_type = str(user_cfg.grpo.loss_type)
    if loss_type == "no_baseline":
        omega_conf_cls.update(config, "algorithm.adv_estimator", ASSIGNMENT_RAW_REWARD_ADV, merge=True)
        omega_conf_cls.update(
            config,
            "actor_rollout_ref.actor.policy_loss.loss_mode",
            ASSIGNMENT_REINFORCE_LOSS,
            merge=True,
        )
    elif loss_type == "reinforce_with_baseline":
        omega_conf_cls.update(config, "algorithm.adv_estimator", ASSIGNMENT_GRPO_ADV, merge=True)
        omega_conf_cls.update(
            config,
            "actor_rollout_ref.actor.policy_loss.loss_mode",
            ASSIGNMENT_REINFORCE_LOSS,
            merge=True,
        )
    elif loss_type == "grpo_clip":
        omega_conf_cls.update(config, "algorithm.adv_estimator", ASSIGNMENT_GRPO_ADV, merge=True)
        omega_conf_cls.update(
            config,
            "actor_rollout_ref.actor.policy_loss.loss_mode",
            ASSIGNMENT_GRPO_CLIP_LOSS,
            merge=True,
        )
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    raw_overrides = omega_conf_cls.to_container(user_cfg.verl.overrides, resolve=True)
    if raw_overrides is None:
        return
    if not isinstance(raw_overrides, dict):
        raise ValueError("verl.overrides must be a mapping from dotted config keys to values.")
    _apply_dot_overrides(config, raw_overrides, omega_conf_cls)


def build_config(verl_stack: dict[str, Any] | None = None) -> Any:
    _assert_exists(DEFAULT_CONFIG_FILE, "Config file")
    _assert_exists(DEFAULT_TRAIN_FILE, "Training parquet")
    _assert_exists(DEFAULT_VALIDATION_FILE, "Validation parquet")
    _assert_exists(REWARD_FILE, "Reward file")

    if verl_stack is None:
        verl_stack = _import_verl_stack()
    omega_conf_cls = verl_stack["OmegaConf"]
    user_cfg = _load_user_config(omega_conf_cls)

    config = _load_base_ppo_config(verl_stack)
    _apply_common_overrides(config, user_cfg, omega_conf_cls)
    config = verl_stack["migrate_legacy_reward_impl"](config)
    verl_stack["auto_set_device"](config)
    return config


def main() -> None:
    args = parse_args()
    verl_stack = _import_verl_stack()
    config = build_config(verl_stack=verl_stack)
    omega_conf_cls = verl_stack["OmegaConf"]

    if args.print_config or args.dry_run:
        print(omega_conf_cls.to_yaml(config, resolve=True))

    if args.dry_run:
        return

    verl_stack["run_ppo"](config)


if __name__ == "__main__":
    main()
