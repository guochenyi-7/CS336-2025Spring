import torch

from cs336_basics.attention import ScaledDotProductAttention
from cs336_basics.transformerLM import TransformerLM
from cs336_basics.utils import cross_entropy_loss
from cs336_basics.optimizer import MyAdamW

def run_benchmark(model, Q, K, V):
    # 预热
    for _ in range(10):
        out = model(Q, K, V)
        out.sum().backward()
        Q.grad = None; K.grad = None; V.grad = None
    torch.cuda.synchronize()
    # 测量前向传播
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    
    start_evt.record()
    out = None
    for _ in range(100):
        out = model(Q, K, V)
    end_evt.record()
    torch.cuda.synchronize()
    fwd_time = start_evt.elapsed_time(end_evt) / 100
    del out 
    torch.cuda.empty_cache()
    # 测量峰值显存
    torch.cuda.reset_peak_memory_stats()
    out = model(Q, K, V)
    mem_bytes = torch.cuda.max_memory_allocated()
    mem_mb = mem_bytes / (1024 ** 2)

    out.sum().backward() 
    Q.grad = None; K.grad = None; V.grad = None
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    # 测量反向传播
    bwd_times = []
    for _ in range(100):
        out = model(Q, K, V)
        loss = out.sum()
        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        loss.backward()
        end_evt.record()
        end_evt.synchronize()
        bwd_times.append(start_evt.elapsed_time(end_evt))
        Q.grad = None; K.grad = None; V.grad = None
    
    avg_bwd_time = sum(bwd_times) / len (bwd_times)

    return fwd_time, avg_bwd_time, mem_mb

    
def benchmark_attention():
    attention = ScaledDotProductAttention()
    attention.cuda()
    compiled_attention = torch.compile(attention)

    batch_size = 8
    d_model_list = [16, 32, 64, 128]
    seq_len_list = [256, 1024, 4096, 8192, 16384]
    header = f"{'d_model':<8} {'seq_len':<8} | {'Van Fwd':<9} {'Cmp Fwd':<9} | {'Van Bwd':<9} {'Cmp Bwd':<9} | {'Van Mem':<9} {'Cmp Mem':<9}"
    print(header)
    print("-" * len(header))
    for d_k in d_model_list:
        for seq_len in seq_len_list:
            try:
                Q = torch.randn(batch_size, seq_len, d_k, device='cuda', dtype=torch.float32, requires_grad=True)
                K = torch.randn(batch_size, seq_len, d_k, device='cuda', dtype=torch.float32, requires_grad=True)
                V = torch.randn(batch_size, seq_len, d_k, device='cuda', dtype=torch.float32, requires_grad=True)

                # 测试普通版
                van_fwd, van_bwd, van_mem = run_benchmark(attention, Q, K, V)

                # 测试编译版 (Compiled)
                cmp_fwd, cmp_bwd, cmp_mem = run_benchmark(compiled_attention, Q, K, V)

                # 打印结果
                print(f"{d_k:<8} {seq_len:<8} | {van_fwd:<9.3f} {cmp_fwd:<9.3f} | {van_bwd:<9.3f} {cmp_bwd:<9.3f} | {van_mem:<9.1f} {cmp_mem:<9.1f}")
                
            except torch.cuda.OutOfMemoryError:
                print("OOM error")
                torch.cuda.empty_cache()


def run_end_to_end_benchmark(model, input_indices, targets, optimizer, tag="Vanilla"):
    print(f"  -> Testing {tag}...", end="", flush=True)

    # 预热
    for _ in range(5):
        optimizer.zero_grad()
        logits = model(input_indices)
        loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    for _ in range(5):
        with torch.no_grad():
            _ = model(input_indices)
    torch.cuda.synchronize()
    
    # 测量前向传播
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    for _ in range(50):
        with torch.no_grad():
            _ = model(input_indices)
    end_evt.record()
    end_evt.synchronize()
    fwd_time = start_evt.elapsed_time(end_evt) / 50

    # 测量整体
    step_times = []
    for _ in range(50):
        torch.cuda.synchronize()

        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()

        logits = model(input_indices)
        loss = cross_entropy_loss(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        end_evt.record()
        end_evt.synchronize()
        step_times.append(start_evt.elapsed_time(end_evt))
    
    avg_step_time = sum(step_times) / len(step_times)
    
    print(" Done.")
    return fwd_time, avg_step_time


def benchmark_transformerLM():
    # 设置超参数
    vocab_size = 10000
    context_size = 1024
    d_model = 512
    num_layers = 6
    num_heads = 8
    d_ff = 2048
    rope_theta = 10000.0
    device = torch.device("cuda")
    dtype = torch.float32

    batch_size = 8
    
    print(f"Model Config: L={num_layers}, H={num_heads}, D={d_model}, Context={context_size}")

    # 设置模型
    model = TransformerLM(
        vocab_size=vocab_size,
        context_size=context_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device,
        dtype=dtype
    ).to(device)

    # 输入
    input_indices  = torch.randint(0, vocab_size, (batch_size, context_size), device=device)
    # 输出
    targets = torch.randint(0, vocab_size, (batch_size, context_size), device=device)

    # 测量普通版
    optimizer = MyAdamW(model.parameters())
    print("\nStarting Vanilla Benchmark...")
    van_fwd, van_total = run_end_to_end_benchmark(model, input_indices, targets, optimizer, tag="Vanilla")

    # 测量编译版
    print("\nCompiling Model... (This may take a minute)")
    compiled_model = torch.compile(model)
    optimizer_compiled = MyAdamW(model.parameters(), lr=1e-3)
    print("Starting Compiled Benchmark...")
    cmp_fwd, cmp_total = run_end_to_end_benchmark(compiled_model, input_indices, targets, optimizer_compiled, tag="Compiled")

    # 打印结果
    print("\n" + "="*65)
    print(f"{'Metric':<20} | {'Vanilla (ms)':<15} {'Compiled (ms)':<15} {'Speedup':<10}")
    print("-" * 65)
    
    fwd_speedup = van_fwd / cmp_fwd if cmp_fwd > 0 else 0
    total_speedup = van_total / cmp_total if cmp_total > 0 else 0
    
    print(f"{'Forward Pass':<20} | {van_fwd:<15.3f} {cmp_fwd:<15.3f} {fwd_speedup:<10.2f}x")
    print(f"{'Fwd+Bwd+Opt Step':<20} | {van_total:<15.3f} {cmp_total:<15.3f} {total_speedup:<10.2f}x")
    print("="*65)

def main():
    torch.set_float32_matmul_precision('high')
    # benchmark_attention()
    benchmark_transformerLM()

if __name__ == "__main__":
    main()