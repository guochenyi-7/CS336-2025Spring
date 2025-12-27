import torch
import torch.nn as nn
import timeit
import argparse
import numpy as np
from cs336_basics.model import BasicsTransformerLM

# 定义不同模型大小的配置字典，方便调用
MODEL_CONFIGS = {
    "small":  {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

def get_args():
    """
    处理命令行参数，让脚本可以通过命令行灵活配置
    """
    parser = argparse.ArgumentParser(description="CS336 Assignment 2 Benchmarking Script")
    
    # 模型配置参数
    parser.add_argument("--model_size", type=str, default="small", choices=MODEL_CONFIGS.keys(),
                        help="Model size from Table 1 (small, medium, etc.)")
    parser.add_argument("--context_length", type=int, default=128, help="Context length (sequence length)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4 [cite: 61])")
    
    # 运行参数
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warm-up steps [cite: 83]")
    parser.add_argument("--measure_steps", type=int, default=10, help="Number of steps to measure [cite: 83]")
    parser.add_argument("--mode", type=str, default="fwd", choices=["fwd", "fwd_bwd"],
                        help="Mode: 'fwd' (forward only) or 'fwd_bwd' (forward + backward)")
    
    return parser.parse_args()

def benchmark():
    # 获取参数
    args = get_args()
    config = MODEL_CONFIGS[args.model_size]
    
    print(f"Running benchmark with config: {args.model_size}, Mode: {args.mode}")
    print(f"Hyperparameters: {config}")

    # 设置设备
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps" 
    print(f"Using device: {device}")
    
    # 初始化模型
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=args.context_length,
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        rope_theta=10000.0,
    )
    model.to(device)
    model.train()

    # 生成随机数据
    # 输入 x 是整数索引，范围在 0 到 vocab_size 之间
    x = torch.randint(0, 10000, (args.batch_size, args.context_length), device=device)
    # 如果需要计算 Loss，我们需要目标 target (对于语言模型通常是 x 的移位，这里为了测速随机生成即可)
    target = torch.randint(0, 10000, (args.batch_size, args.context_length), device=device)
    criterion = torch.nn.CrossEntropyLoss()
    # 定义运行一步的函数
    def run_step():
        # 前向传播
        logits = model(x)
        
        if args.mode == "fwd_bwd":
            # 如果需要反向传播，我们需要计算一个 Loss
            loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
            loss.backward()
            
            # 每次反向传播后清空梯度，防止累积占用显存或影响计算
            model.zero_grad()

        # 等待 GPU 完成所有计算
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize() # 如果你以后开启了 Mac 的 GPU 加速 (MPS)

    # 预热 (Warm-up)
    print(f"Starting {args.warmup_steps} warm-up steps...")
    for _ in range(args.warmup_steps):
        run_step()

    # 正式计时
    print(f"Measuring {args.measure_steps} steps...")
    times = []
    
    for _ in range(args.measure_steps):
        # 记录开始时间
        start_time = timeit.default_timer()
        
        run_step()
        
        # 记录结束时间
        end_time = timeit.default_timer()
        times.append(end_time - start_time)

    # 计算统计结果
    times_np = np.array(times)
    avg_time = np.mean(times_np)
    std_time = np.std(times_np)

    print("-" * 30)
    print(f"times: {[f'{t:.6f}' for t in times]}")
    print(f"Results for {args.model_size} ({args.mode}):")
    print(f"Average time: {avg_time:.6f} s")
    print(f"Std Dev:      {std_time:.6f} s")
    print("-" * 30)

if __name__ == "__main__":
    benchmark()