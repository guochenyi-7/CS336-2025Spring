import numpy as np
import wandb
from typing import Callable, List, Dict

def log_generations(
    prompts: List[str],
    responses: List[str],
    ground_truths: List[str],
    reward_fn: Callable[[str, str], Dict[str, float]],
    entropies: List[float],
    step: int,
    prefix: str = "eval"
) -> None:
    """
    记录模型生成的回复及其相关指标到 wandb。
    """
    format_rewards = []
    answer_rewards = []
    total_rewards = []
    
    lengths = []
    correct_lengths = []
    incorrect_lengths = []

    # 用于 wandb 记录的可视化表格
    table_data = []

    for prompt, response, gt, entropy in zip(prompts, responses, ground_truths, entropies):
        # 计算奖励信息
        reward_dict = reward_fn(response, gt)
        f_reward = reward_dict.get("format_reward", 0.0)
        a_reward = reward_dict.get("answer_reward", 0.0)
        t_reward = reward_dict.get("reward", 0.0)

        format_rewards.append(f_reward)
        answer_rewards.append(a_reward)
        total_rewards.append(t_reward)

        # 计算回复长度
        # 这里使用字符长度，如果有 tokenizer，也可以替换为 len(tokenizer.encode(response))
        length = len(response) 
        lengths.append(length)
        
        if a_reward > 0.0:  # 假设 answer_reward 大于 0 即为正确
            correct_lengths.append(length)
        else:
            incorrect_lengths.append(length)

        # 汇总单条数据用于表格展示
        table_data.append([
            prompt, 
            response, 
            gt, 
            f_reward, 
            a_reward, 
            t_reward, 
            entropy, 
            length
        ])

    # 创建 Wandb 表格
    table = wandb.Table(
        columns=[
            "Prompt", "Response", "Ground Truth", 
            "Format Reward", "Answer Reward", "Total Reward", 
            "Entropy", "Length"
        ],
        data=table_data
    )

    # 计算各项平均值并记录
    metrics = {
        f"{prefix}/mean_format_reward": np.mean(format_rewards) if format_rewards else 0.0,
        f"{prefix}/mean_answer_reward": np.mean(answer_rewards) if answer_rewards else 0.0,
        f"{prefix}/mean_total_reward": np.mean(total_rewards) if total_rewards else 0.0,
        f"{prefix}/mean_entropy": np.mean(entropies) if entropies else 0.0,
        f"{prefix}/mean_length": np.mean(lengths) if lengths else 0.0,
        f"{prefix}/mean_correct_length": np.mean(correct_lengths) if correct_lengths else 0.0,
        f"{prefix}/mean_incorrect_length": np.mean(incorrect_lengths) if incorrect_lengths else 0.0,
        f"{prefix}/generations": table
    }

    # 推送到 Wandb
    wandb.log(metrics, step=step)
    