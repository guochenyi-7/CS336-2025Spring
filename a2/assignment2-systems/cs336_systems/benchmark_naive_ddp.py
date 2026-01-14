import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd

from cs336_basics.transformerLM import TransformerLM

MODEL_CONFIG_XL = {
    "vocab_size": 10000,
    "context_size": 128,
    "d_model": 1600,
    "d_ff": 6400,
    "num_layers": 48,
    "num_heads": 25,
    "rope_theta": 10000,
}

MODEL_CONFIG_S = {
    "vocab_size": 200,
    "context_size": 64,
    "d_model": 160,
    "d_ff": 50,
    "num_layers": 2,
    "num_heads": 2,
    "rope_theta": 10000,
}

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

def run_benchmark_process(rank, world_size, args):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42 + rank)

    model = TransformerLM(**MODEL_CONFIG_XL)
    model = model.to(device)

    # 广播初始权重
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    # 随机数据
    batch_size = 4
    vocab_size = MODEL_CONFIG_XL["vocab_size"]
    context_len = MODEL_CONFIG_XL["context_size"]
    
    # 预生成一批数据
    x = torch.randint(0, vocab_size, (batch_size, context_len)).to(device)
    y = torch.randint(0, vocab_size, (batch_size * context_len,)).to(device) #Flatten targets
    
    # 预热 
    for _ in range(3):
        optimizer.zero_grad()
        output = model(x) 
        output = output.view(-1, vocab_size)
        loss = loss_fn(output, y)
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size
        optimizer.step()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if rank == 0:
        print("Warmup finished. Starting benchmark...")
        
    # 正式测量
    num_steps = 10
    total_times = []
    comm_times = []
    
    for step in range(num_steps):
        step_start = time.time() # CPU 时间 (Python overhead 也是重点)
        
        optimizer.zero_grad()
        output = model(x)
        output = output.view(-1, vocab_size)
        loss = loss_fn(output, y)
        loss.backward()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize() # 等待反向传播计算完成
        
        # --- 测量通信时间 ---
        comm_start = time.time() 
        for param in model.parameters():
            if param.grad is not None:
                # Naive DDP: 单个参数逐一通信
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size
        if torch.cuda.is_available():
            torch.cuda.synchronize() # 等待所有通信完成
        comm_end = time.time()
        # -------------------
        
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_end = time.time()
        
        total_times.append((step_end - step_start) * 1000) # 转为 ms
        comm_times.append((comm_end - comm_start) * 1000) # 转为 ms
        
    # 报告结果
    if rank == 0:
        avg_total = sum(total_times) / len(total_times)
        avg_comm = sum(comm_times) / len(comm_times)
        comm_ratio = (avg_comm / avg_total) * 100
        
        print("\n=== Naive DDP Benchmark Results (XL Model) ===")
        print(f"Config: {MODEL_CONFIG_XL}")
        print(f"Num GPUs: {world_size}")
        print(f"Avg Total Time per Step: {avg_total:.2f} ms")
        print(f"Avg Communication Time:  {avg_comm:.2f} ms")
        print(f"Communication Overhead:  {comm_ratio:.2f}%")
        
    dist.destroy_process_group()

def benchmark_naive_ddp():
    world_size = 2
    mp.spawn(
        run_benchmark_process,
        args=(world_size, None),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    benchmark_naive_ddp()