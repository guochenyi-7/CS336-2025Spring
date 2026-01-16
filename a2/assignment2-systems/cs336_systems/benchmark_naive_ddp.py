import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd

from cs336_basics.transformerLM import TransformerLM
from cs336_systems.ddp_overlap_individual_parameters import DdpOverlapIndividualParameters
from cs336_systems.bucketed_ddp import BucketedDDP
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

MODEL_CONFIG_XL = {
    "vocab_size": 50257,
    "context_size": 1024,
    "d_model": 1200,      
    "d_ff": 4800,           
    "num_layers": 10,
    "num_heads": 24,
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

def distributed_each(model):
     world_size = dist.get_world_size()
     for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size

def distributed_all(model):
    world_size = dist.get_world_size()
    grads = [param.grad for param in model.parameters() if param.grad is not None]
        
    if len(grads) > 0:
        # 将所有梯度打平成一个连续的大张量
        flat_grads = _flatten_dense_tensors(grads)
        
        # 仅发起一次 all-reduce 调用
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
        flat_grads /= world_size
        
        # 将同步后的梯度写回原张量
        # _unflatten_dense_tensors 返回的是新张量列表，需要拷贝回 grads
        synced_grads = _unflatten_dense_tensors(flat_grads, grads)
        for old_grad, new_grad in zip(grads, synced_grads):
            old_grad.copy_(new_grad)


def run_benchmark_process(rank, world_size, args):
    setup(rank, world_size)
    
    # 从 args 中获取配置
    mode = args.get('mode', 'naive')
    bucket_size_mb = args.get('bucket_size_mb', 25.0)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    # 保证每个进程的初始权重和数据分布在测量中具有一致性
    torch.manual_seed(42) 

    # 初始化模型
    raw_model = TransformerLM(**MODEL_CONFIG_XL).to(device)
    
    # 根据模式包装模型
    if mode == 'overlap':
        model = DdpOverlapIndividualParameters(raw_model)
    elif mode == 'bucketed':
        model = BucketedDDP(raw_model, bucket_size_mb=bucket_size_mb)
    else:
        # 朴素模式：手动广播
        model = raw_model
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    # 3. 准备随机数据
    x = torch.randint(0, MODEL_CONFIG_XL["vocab_size"], (4, MODEL_CONFIG_XL["context_size"])).to(device)
    y = torch.randint(0, MODEL_CONFIG_XL["vocab_size"], (4 * MODEL_CONFIG_XL["context_size"],)).to(device)
    
    # --- 预热阶段 (Warmup) ---
    for _ in range(5):
        optimizer.zero_grad()
        output = model(x).view(-1, MODEL_CONFIG_XL["vocab_size"])
        loss = loss_fn(output, y)
        loss.backward()
        
        if mode in ['overlap', 'bucketed']:
            model.finish_gradient_synchronization()
        else:
            distributed_all(model)
        optimizer.step()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if rank == 0:
        print(f"[{mode.upper()}] Warmup finished. Starting benchmark...")
        
    # --- 正式测量阶段 ---
    num_steps = 10
    total_step_times = []
    bw_comm_times = [] # 测量反向传播 + 梯度同步的总耗时
    
    for step in range(num_steps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_start = time.time()
        
        # --- 前向传播 ---
        optimizer.zero_grad()
        output = model(x).view(-1, MODEL_CONFIG_XL["vocab_size"])
        loss = loss_fn(output, y)
        
        # --- 反向传播 + 通信 ---
        bw_start = time.time()
        loss.backward()
        
        if mode in ['overlap', 'bucketed']:
            # 等待所有后端的 all-reduce handle 完成，并进行梯度平均
            model.finish_gradient_synchronization() 
        else:
            # Naive 模式：必须先同步保证 Backward 结束，再发起同步
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            distributed_all(model) 

        if torch.cuda.is_available():
            torch.cuda.synchronize() # 确保所有 GPU 操作完成
        bw_end = time.time()
        
        # --- 优化器更新 ---
        optimizer.step()
        
        step_end = time.time()
        
        total_step_times.append((step_end - step_start) * 1000)
        bw_comm_times.append((bw_end - bw_start) * 1000)
        
    # 汇总结果
    if rank == 0:
        avg_total = sum(total_step_times) / len(total_step_times)
        avg_bw_comm = sum(bw_comm_times) / len(bw_comm_times)
        print(f"\n=== {mode.upper()} DDP Results ===")
        if mode == 'bucketed':
            print(f"Bucket Size: {bucket_size_mb} MB")
        print(f"Avg Total Step Time: {avg_total:.2f} ms")
        print(f"Avg (Backward + Comm) Time: {avg_bw_comm:.2f} ms")
        
    dist.destroy_process_group()

def benchmark_all_strategies():
    world_size = 2
    test_configs = [
        {'mode': 'naive'},
        {'mode': 'overlap'},
        {'mode': 'bucketed', 'bucket_size_mb': 1.0},
        {'mode': 'bucketed', 'bucket_size_mb': 10.0}, 
        {'mode': 'bucketed', 'bucket_size_mb': 100.0}, 
        {'mode': 'bucketed', 'bucket_size_mb': 1000.0}, 
    ]
    
    for config in test_configs:
        mp.spawn(
            run_benchmark_process,
            args=(world_size, config),
            nprocs=world_size,
            join=True
        )

if __name__ == "__main__":
    benchmark_all_strategies()