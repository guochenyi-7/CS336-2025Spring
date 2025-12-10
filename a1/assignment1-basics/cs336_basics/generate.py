import torch
import argparse
import pickle
import numpy as np

from .transformerLM import TransformerLM
from .tokenizer import Tokenizer
from .utils import load_checkpoint
from .softmax import softmax

def Generate(
        model,
        prompt_token_ids,
        max_new_tokens,
        eos_id,
        temperature,
        top_p,
):
    model.eval()
    cur_tokens_id = prompt_token_ids.clone()

    # 完成掩码
    finished_mask = torch.zeros(cur_tokens_id.shape[0], 1, dtype=torch.bool, device=cur_tokens_id.device)
    eos_tensor = torch.tensor(eos_id, device=prompt_token_ids.device)

    for _ in range(max_new_tokens):
        # 前向传播
        with torch.no_grad():
            logits = model(cur_tokens_id)
        
        # 获取最后一个时间步的预测
        # [batch_size, vocab_size]
        next_token_ids = logits[:, -1, :]

        # 应用温度
        if temperature > 0:
            next_token_ids = next_token_ids / temperature
        else:
            next_token_ids = next_token_ids / 1e-8

        probs = softmax(next_token_ids, dim=-1)
        if top_p < 1.0:
            # 排序
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            # 前缀和
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # 创建移除掩码，注意边界词也被标记要移除了
            sorted_indices_to_remove = cumulative_probs > top_p
            # 移位，把边界词包含进来
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            # 将掩码映射回去
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            # 将被移除的词的概率设为0
            probs[indices_to_remove] = 0.0
            # 重新归一化
            probs = probs / probs.sum(dim=-1, keepdim=True)

        # 抽样
        next_token_id = torch.multinomial(probs, num_samples=1)
        next_token_id = torch.where(finished_mask, eos_tensor, next_token_id)
        finished_mask = finished_mask | (next_token_id == eos_id)
       
        # 拼接
        cur_tokens_id = torch.cat([cur_tokens_id, next_token_id], dim=1)
        
        if finished_mask.all():
            break
    
    return cur_tokens_id

def main(args):
    # 设置设备
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    #  加载分词器 (Tokenizer)
    # 我们之前用 pickle 保存了 tokenizer_model.pkl，现在读出来重建 Tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    with open(args.tokenizer_path, 'rb') as f:
        tokenizer_data = pickle.load(f)
    
    tokenizer = Tokenizer(
        vocab=tokenizer_data["vocab"],
        merges=tokenizer_data["merges"],
        special_tokens=tokenizer_data["special_tokens"]
    )

    # 初始化模型结构 (必须与训练时参数一致)
    # 注意：这里我们手动填入训练时的参数，或者你可以把这些参数也保存到 checkpoint 里
    model = TransformerLM(
        vocab_size=10000,
        context_size=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000.0,
    )
    model.to(device)

    # 加载模型权重 (Checkpoint)
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 准备 Prompt
    prompt_text = args.prompt
    print(f"\nPrompt: \"{prompt_text}\"")
    
    # 编码 Prompt
    prompt_ids = tokenizer.encode(prompt_text)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device) # [1, seq_len]

    # 获取 EOS token ID (通常是 <|endoftext|>)
    eos_id = tokenizer.special_tokens.index("<|endoftext|>") + 256 + len(tokenizer.merges)
    print(f"eos_id: {eos_id}")
    # 或者更安全的方法：再编码一次 eos
    # eos_id = tokenizer.encode("<|endoftext|>")[0]

    # 生成
    print("Generating...")
    generated_ids = Generate(
        model=model,
        prompt_token_ids=prompt_tensor,
        max_new_tokens=args.max_new_tokens,
        eos_id=eos_id,
        temperature=args.temperature,
        top_p=args.top_p
    )

    # 解码并打印
    # generated_ids 是 [1, total_len]，转成 list
    generated_list = generated_ids[0].tolist()
    generated_text = tokenizer.decode(generated_list)
    
    print("-" * 40)
    print("Generated Story:")
    print("-" * 40)
    print(generated_text)
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True, help="模型检查点路径 (.pt)")
    parser.add_argument('--tokenizer_path', type=str, default="data/tokenizer_model.pkl", help="分词器路径 (.pkl)")
    parser.add_argument('--prompt', type=str, default="Once upon a time", help="故事的开头")
    parser.add_argument('--max_new_tokens', type=int, default=200, help="最大生成长度")
    parser.add_argument('--temperature', type=float, default=0.8, help="采样温度 (0.0-1.0)")
    parser.add_argument('--top_p', type=float, default=0.9, help="Top-P 采样阈值")
    parser.add_argument('--device', type=str, default="auto")
    
    args = parser.parse_args()
    main(args)