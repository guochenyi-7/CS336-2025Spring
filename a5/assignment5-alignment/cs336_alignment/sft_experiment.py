import torch
import wandb
import json

from torch.nn.utils import clip_grad_norm_
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.get_response_log_probs import get_response_log_probs
from cs336_alignment.sft_microbatch_train_step import sft_microbatch_train_step
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    启动推理过程，这里我们使用 vLLM 将模型保存在一个与策略模型独立的 GPU 上。
    """
    vllm_set_random_seed(seed)
    
    # 猴子补丁（Monkeypatch）来自 TRL：
    # https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # 对 LLM 进行补丁，以确保我们可以：
    # (1) 将 vLLM 模型放置在所需的设备上 (world_size_patch)；并且
    # (2) 避免运行一个并非为我们设置设计的测试 (profiling_patch)。
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

def run_sft_experiment():
    # ==========================================
    # 1. 初始化配置与模型加载
    # ==========================================
    model_id = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    train_device = "cuda:0"
    vllm_device = "cuda:1"
    
    # 初始化 Wandb
    wandb.init(project="cs336-assignment5-sft")
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # 在 GPU 0 加载策略模型用于训练
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(train_device)
    
    # 在 GPU 1 启动 vLLM 用于评估
    vllm_engine = init_vllm(model_id, device=vllm_device, seed=42)
    
    # 采样参数
    eval_sampling_params = SamplingParams(
        temperature=1.0, 
        top_p=1.0, 
        max_tokens=1024, 
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # 优化器与超参数设置
    learning_rate = 1e-5 # 需要你进行微调
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
    gradient_accumulation_steps = 4 # 假设值
    eval_interval = 50 # 每 50 个 step 评估一次

    # ==========================================
    # 2. 数据加载与预处理 (伪代码)
    # ==========================================
    # 读取 /data/a5-alignment/MATH/sft.jsonl
    # 这里你需要将 jsonl 转换为 DataLoader，并在 collate_fn 中使用 tokenize_prompt_and_output
    train_dataloader = get_sft_dataloader(...) 
    val_prompts = get_val_prompts(...) # 读取 validation.jsonl 中的 prompts
    val_ground_truths = get_val_ground_truths(...)

    # ==========================================
    # 3. 主训练循环
    # ==========================================
    global_step = 0
    policy_model.train()
    num_epochs = 100

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
            loss, metadata = sft_microbatch_train_step(
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
                    "train/loss": loss.item(),
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
    policy_model.save_pretrained("/data/yourusername/sft_model")
    tokenizer.save_pretrained("/data/yourusername/sft_model")

if __name__ == "__main__":
    run_sft_experiment()

