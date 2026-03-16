import argparse
import os
os.environ["OMP_NUM_THREADS"] = "8"

import wandb
import json
import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import torch
from torch.nn.utils import clip_grad_norm_
from vllm.model_executor.utils import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from cs336_alignment.get_response_log_probs import get_response_log_probs
from cs336_alignment.sft_microbatch_train_step import sft_microbatch_train_step
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

from torch.utils.data import Dataset, DataLoader
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output

PROMPT_PATH = project_root / "cs336_alignment" / "prompts" / "r1_zero.prompt"
with open(PROMPT_PATH, "r", encoding="utf-8") as file:
    PROMPT_TEMPLATE = file.read()

DATASET_CONFIGS = {
    "gsm8k": {
        "train_data_path": project_root / "data" / "gsm8k" / "sft_formatted.jsonl",
        "val_data_path": project_root / "data" / "gsm8k" / "test.jsonl",
        "question_key": "question",
    },
    "math": {
        "train_data_path": project_root / "data" / "MATH" / "sft.jsonl",
        "val_data_path": project_root / "data" / "MATH" / "validation.jsonl",
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

class SFTDataset(Dataset):
    def __init__(self, data_path, num_samples=None):
        """
        读取已经格式化好的 SFT jsonl 文件，要求至少包含 prompt/response 字段。
        """
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
                if num_samples and len(self.data) >= num_samples:
                    break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def create_collate_fn(tokenizer):
    """
    返回一个可用于 DataLoader 的 collate 函数，内部调用 tokenize_prompt_and_output
    """
    def collate_fn(batch):
        # 提取 batch 中的 prompt 和 response
        prompt_strs = [item["prompt"] for item in batch]
        output_strs = [item["response"] for item in batch]

        # 调用你作业中要求实现的 tokenize 函数
        # 它应该返回形如 {"input_ids": ..., "labels": ..., "response_mask": ...} 的字典
        tokenized_batch = tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
        return tokenized_batch
    
    return collate_fn

def load_val_data(val_file_path, question_key):
    prompts = []
    ground_truths = []
    with open(val_file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            prompts.append(PROMPT_TEMPLATE.format(question=item[question_key]))
            ground_truths.append(extract_ground_truth(item["answer"]))
    return prompts, ground_truths

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.8):
    """启动推理过程，使用 vLLM 将模型保存在一个与策略模型独立的 GPU 上。"""
    vllm_set_random_seed(seed)
    # 猴子补丁（Monkeypatch）来自 TRL：
    # https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """复制自 https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670"""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def run_sft_experiment(dataset_name: str = "math", num_train_samples: Optional[int] = 1024):
    # ==========================================
    # 1. 初始化配置与模型加载
    # ==========================================
    model_id = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    train_device = "cuda:0"
    vllm_device = "cuda:1"
    dataset_config = get_dataset_config(dataset_name)
    
    # 初始化 Wandb
    wandb.init(
        project="cs336-assignment5-sft",
        config={
            "dataset": dataset_name.lower(),
            "train_data_path": str(dataset_config["train_data_path"]),
            "val_data_path": str(dataset_config["val_data_path"]),
            "num_train_samples": num_train_samples,
        },
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # 在 GPU 0 加载策略模型用于训练
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa"
    ).to(train_device)
    policy_model.gradient_checkpointing_enable()
    policy_model.config.use_cache = False
    # 在 GPU 1 启动 vLLM 用于评估
    vllm_engine = init_vllm(model_id, device=vllm_device, seed=42)
    # 采样参数
    eval_sampling_params = SamplingParams(
        temperature=0.0, 
        top_p=1.0, 
        max_tokens=256, 
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # 优化器与超参数设置
    learning_rate = 1e-5
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
    train_batch_size = 8
    gradient_accumulation_steps = 2
    eval_interval = 50 # 每 50 个 step 评估一次

    # ==========================================
    # 2. 数据加载与预处理
    # ==========================================
    train_data_path = str(dataset_config["train_data_path"])
    val_data_path = str(dataset_config["val_data_path"])

    train_dataset = SFTDataset(train_data_path, num_samples=num_train_samples)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size, 
        shuffle=True,
        collate_fn=create_collate_fn(tokenizer)
    )

    # 加载验证集数据，用于 eval_step 时传给 evaluate_vllm
    val_prompts, val_ground_truths = load_val_data(
        val_data_path,
        question_key=dataset_config["question_key"],
    )

    # ==========================================
    # 3. 主训练循环
    # ==========================================
    global_step = 0
    policy_model.train()
    num_epochs = 10

    for epoch in range(num_epochs):
        for idx, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(train_device)
            labels = batch["labels"].to(train_device)
            response_mask = batch["response_mask"].to(train_device)

            # 获取 log_probs
            log_prob_dict = get_response_log_probs(
                model=policy_model, 
                input_ids=input_ids, 
                labels=labels, 
                return_token_entropy=True
            )
            policy_log_probs = log_prob_dict["log_probs"]

            # 计算 loss 并反向传播 (已在内部除以 gradient_accumulation_steps)
            scaled_loss, metadata = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps
            )

            # 梯度累加逻辑
            if (idx + 1) % gradient_accumulation_steps == 0:
                # 梯度裁剪
                clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # 记录训练指标
                wandb.log({
                    "train_step": global_step,
                    "train/loss": metadata["unscaled_loss"].item(), 
                    "train/entropy": log_prob_dict["token_entropy"].mean().item()
                })

                # ==========================================
                # 4. 定期评估逻辑
                # ==========================================
                if global_step % eval_interval == 0:
                    policy_model.eval()
                    
                    # 热同步权重到 GPU 1 的 vLLM 中 
                    load_policy_into_vllm_instance(policy_model, vllm_engine)
                    
                    # 使用 vLLM 生成回答
                    outputs = vllm_engine.generate(val_prompts, eval_sampling_params)
                    
                    # 计算准确率
                    correct_count = 0
                    for output, truth in zip(outputs, val_ground_truths):
                        generated_text = output.outputs[0].text
                        # 使用提供的奖励函数
                        reward_dict = r1_zero_reward_fn(generated_text, truth)
                        if reward_dict["answer_reward"] == 1.0:
                            correct_count += 1
                            
                    val_accuracy = correct_count / len(val_prompts)
                    
                    wandb.log({
                        "eval_step": global_step,
                        "eval/accuracy": val_accuracy
                    })
                    
                    print(f"Step {global_step} | Validation Accuracy: {val_accuracy:.2%}")
                    policy_model.train() # 切回训练模式

    # 训练结束后保存模型
    save_dir = project_root / "outputs" / "sft_model"
    save_dir.mkdir(parents=True, exist_ok=True)

    policy_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_CONFIGS),
        default="math",
        help="Dataset to use for SFT training and validation.",
    )
    parser.add_argument(
        "--num-train-samples",
        type=int,
        default=1024,
        help="Optional cap on the number of training samples to load.",
    )
    args = parser.parse_args()
    run_sft_experiment(dataset_name=args.dataset, num_train_samples=args.num_train_samples)
