import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os

from cs336_basics.transformerLM import TransformerLM
from cs336_systems.sharded_optimizer import ShardedOptimizer

# --- 配置部分 ---
MODEL_CONFIG_XL = {
    "vocab_size": 50257,
    "context_size": 1024,
    "d_model": 1200,      
    "d_ff": 4800,           
    "num_layers": 10,
    "num_heads": 24,
    "rope_theta": 10000,
}

def get_memory_mb():
    """获取当前 GPU 实际分配的显存（MB）"""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024**2)

def run_benchmark(rank, world_size, use_sharding):
    # 环境初始化
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 强制清理缓存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 2. 初始化模型
    model = TransformerLM(**MODEL_CONFIG_XL).to(device)
    
    # 广播权重确保同步
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # 封装 DDP (ShardedOptimizer 仍需 DDP 处理梯度的 All-Reduce)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # 记录初始化后内存 (Model Weights)
    mem_init = get_memory_mb()

    # 初始化优化器
    # 注意：AdamW 在初始时不会分配状态内存，只有第一次 step 时才会分配
    if use_sharding:
        optimizer = ShardedOptimizer(model.parameters(), optim.AdamW, lr=1e-3)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # 构造输入并运行 Forward/Backward
    batch_size = 4
    input_ids = torch.randint(
        0, 
        MODEL_CONFIG_XL["vocab_size"], 
        (batch_size, MODEL_CONFIG_XL["context_size"]), 
        device=device,
        dtype=torch.long
    )

    logits = model(input_ids)
    loss = logits.sum()
    loss.backward()

    # 清理激活值干扰
    # 为了准确测量优化器状态，删除不再需要的 logits 和 loss，
    # 并调用 empty_cache 释放 backward 过程中产生的临时峰值显存。
    del logits
    del loss
    torch.cuda.empty_cache() 

    # 记录优化器 Step 之前的当前内存 (Model Weights + Gradients)
    mem_before_step = get_memory_mb()

    # 运行优化器 Step (此时分配 Optimizer States)
    start_time = time.time()
    optimizer.step()
    torch.cuda.synchronize() 
    step_time = time.time() - start_time

    # 记录优化器 Step 之后的当前内存 (Weights + Gradients + Opt States)
    mem_after_step = get_memory_mb()

    # 汇总输出
    if rank == 0:
        mode = "With Sharding" if use_sharding else "No Sharding (Baseline)"
        print("="*60)
        print(f"Running Mode: {mode}")
        print(f"Model Config: {MODEL_CONFIG_XL}")
        print("-" * 30)
        print(f"Current Memory [1. Init] (Weights):           {mem_init:.2f} MB")
        print(f"Current Memory [2. Before Step] (+ Grads):    {mem_before_step:.2f} MB")
        print(f"Current Memory [3. After Step] (+ Opt States): {mem_after_step:.2f} MB")
        print("-" * 30)
        print(f"Step Time: {step_time:.4f} s")
        
        mem_grads = mem_before_step - mem_init
        mem_opt = mem_after_step - mem_before_step
        print("-" * 30)
        print(f"Approx Gradient Size:       {mem_grads:.2f} MB")
        print(f"Approx Optimizer State Size:{mem_opt:.2f} MB")
        
        if use_sharding:
             print(f"\n[Analysis] With World_Size={world_size}, Opt State should be ~1/{world_size} of Baseline.")
        print("="*60 + "\n")

    dist.destroy_process_group()

if __name__ == "__main__":
    WORLD_SIZE = 2 
    
    # 运行 Baseline
    print(f"Starting Benchmark: Baseline (World Size: {WORLD_SIZE})")
    mp.spawn(run_benchmark, args=(WORLD_SIZE, False), nprocs=WORLD_SIZE, join=True)
    
    time.sleep(2) 

    # 运行 Sharded Optimizer
    print(f"Starting Benchmark: Sharded Optimizer (World Size: {WORLD_SIZE})")
    mp.spawn(run_benchmark, args=(WORLD_SIZE, True), nprocs=WORLD_SIZE, join=True)