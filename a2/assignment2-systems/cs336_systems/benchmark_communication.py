import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import argparse

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    # 根据后端初始化
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # 如果是 NCCL，必须绑定当前进程到指定的 GPU
    if backend == "nccl":
        torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def run_benchmark(rank, world_size, args):
    setup(rank, world_size, args.backend)
    
    # 准备数据
    # 1MB = 1024 * 1024 字节
    # float32 占 4 字节，所以元素数量 = 字节数 / 4
    num_elements = int((args.size_mb * 1024 * 1024) / 4)
    
    if args.backend == "nccl":
        device = torch.device(f"cuda:{rank}")
        # 在 GPU 上创建数据
        data = torch.randn(num_elements, device=device, dtype=torch.float32)
    else:
        device = torch.device("cpu")
        # 在 CPU 上创建数据
        data = torch.randn(num_elements, device=device, dtype=torch.float32)

    # 预热
    # 对 NCCL 尤为重要，建立通信缓存
    for _ in range(5):
        dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
    
    if args.backend == "nccl":
        torch.cuda.synchronize()

    # 正式计时
    timings = []
    iterations = 10 # 测量 10 次取平均
    
    for _ in range(iterations):
        # 确保所有进程在这里同步开始
        dist.barrier()
        
        if args.backend == "nccl":
            # GPU 计时方式
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
            end_event.record()
            
            # 等待 GPU 完成
            end_event.synchronize() 
            elapsed_time_ms = start_event.elapsed_time(end_event) # 返回毫秒
            timings.append(elapsed_time_ms / 1000.0) # 转换为秒
            
        else:
            # CPU 计时方式
            start_time = time.perf_counter()
            dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
            end_time = time.perf_counter()
            timings.append(end_time - start_time)

    # 汇总结果
    mean_time = sum(timings) / len(timings)
    
    # 只让 Rank 0 打印结果，避免刷屏
    if rank == 0:
        print(f"[{args.backend.upper()}] | Processes: {world_size} | "
              f"Data Size: {args.size_mb} MB | "
              f"Avg Time: {mean_time:.6f} s | "
              f"Bandwidth: {(args.size_mb / mean_time):.2f} MB/s")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="gloo", choices=["gloo", "nccl"], 
                        help="Backend to use: gloo for CPU, nccl for GPU")
    parser.add_argument("--size_mb", type=float, default=10, 
                        help="Size of tensor in MB")
    parser.add_argument("--world_size", type=int, default=2, 
                        help="Number of processes")
    args = parser.parse_args()

    # 硬件检查
    if args.backend == "nccl" and args.world_size > torch.cuda.device_count():
        print(f"Warning: You requested {args.world_size} processes for NCCL but only have {torch.cuda.device_count()} GPUs.")
        print("This is not supported for standard DDP benchmarks.")
        exit(1)

    mp.spawn(run_benchmark,
             args=(args.world_size, args),
             nprocs=args.world_size,
             join=True)