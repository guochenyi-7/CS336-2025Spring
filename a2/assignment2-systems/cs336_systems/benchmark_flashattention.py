import torch
import triton
import pandas as pd
import math
import os

# 检查是否导入成功
from cs336_systems.flashattention_with_triton import FlashattentionWithTriton
from cs336_basics.attention import ScaledDotProductAttention


# -----------------------------------------------------------------------------
# 配置参数
# -----------------------------------------------------------------------------
# 序列长度扫描范围：从 128 (2^7) 到 32768 (2^15)
# 注意：在 24GB 显存上，PyTorch 原生 Attention 在 16384 或 32768 时极大概率 OOM。
# 如果你想强行测 65536，可以把 16 改回 17，但 Triton 可能能跑，PyTorch 必挂。
SEQ_LENS = [2**i for i in range(7, 16)]  # [128, ..., 32768]

# Head 维度扫描
HEAD_DIMS = [64, 128]  # 4090D 上常用的维度，简化扫描范围以节省时间

# 精度扫描
# 4090D 完美支持 bfloat16，建议重点关注 bf16
DTYPES = [torch.float32, torch.bfloat16]

BATCH_SIZE = 1
IS_CAUSAL = True

# -----------------------------------------------------------------------------
# PyTorch Baseline 实现
# -----------------------------------------------------------------------------
def manual_attention_reference(q, k, v, is_causal=False):
    # (B, L, D) -> (B, H=1, L, D) 模拟单头，或者直接按 (B, L, D) 计算
    # 这里保持输入维度 (1, L, D) 进行计算
    scale = 1.0 / math.sqrt(q.shape[-1])
    
    # 显存优化技巧：在 PyTorch 2.0+ 上，如果单纯测算子性能，不保留梯度图可以跑更大 Batch
    # 但为了对比 Backward 性能，这里必须保留计算图
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    if is_causal:
        seq_len = q.shape[1]
        # 使用 register_buffer 风格的 mask 创建方式防止重复开销，这里简化为现场创建
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, float('-inf'))
    
    attn_probs = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, v)
    return output

# -----------------------------------------------------------------------------
# Benchmark 核心函数
# -----------------------------------------------------------------------------
def run_benchmark(func, q, k, v, is_causal, grad_output, provider):
    """
    运行 benchmark 并返回 (fwd, bwd, e2e) 时间 (ms)
    """
    # 垃圾回收，防止显存碎片影响 4090D 这种显存较紧凑的卡
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Forward Benchmark
    try:
        # 4090D 算力较弱，减少 quantile 统计的开销，rep 稍微降低
        fwd_ms = triton.testing.do_bench(lambda: func(q, k, v, is_causal), warmup=10, rep=50)
    except torch.cuda.OutOfMemoryError:
        return float('inf'), float('inf'), float('inf')
    except Exception as e:
        print(f"[{provider} FWD Error] {e}")
        return float('nan'), float('nan'), float('nan')

    # E2E (Fwd + Bwd) Benchmark
    def fwd_bwd_pass():
        # 重新清空梯度
        if q.grad is not None: q.grad = None
        if k.grad is not None: k.grad = None
        if v.grad is not None: v.grad = None
        
        o = func(q, k, v, is_causal)
        o.backward(grad_output, retain_graph=True)
    
    try:
        gc.collect()
        torch.cuda.empty_cache()
        e2e_ms = triton.testing.do_bench(fwd_bwd_pass, grad_to_none=[q, k, v], warmup=10, rep=50)
    except torch.cuda.OutOfMemoryError:
        return fwd_ms, float('inf'), float('inf')
    except Exception as e:
        print(f"[{provider} BWD Error] {e}")
        return fwd_ms, float('nan'), float('nan')

    bwd_ms = e2e_ms - fwd_ms
    return fwd_ms, bwd_ms, e2e_ms

def pytorch_wrapper(q, k, v, is_causal):
    # 每次调用实例化一个新的 model (或者你可以把它放在外面，但这里开销很小)
    model = ScaledDotProductAttention()
    
    # 根据 is_causal 构造 mask
    mask = None
    if is_causal:
        seq_len = q.shape[-2]
        # 创建下三角 mask
        mask = torch.tril(torch.ones((seq_len, seq_len), device=q.device)).bool()
    
    # 调用模型并返回结果
    return model(q, k, v, mask)

# -----------------------------------------------------------------------------
# 主程序
# -----------------------------------------------------------------------------
def main():
    # 设置 PyTorch 显存分配策略，减少碎片化（对 24GB 卡跑大模型很有用）
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    device = torch.device("cuda")
    torch.manual_seed(0)

    print(f"Running on: {torch.cuda.get_device_name(0)}")
    print(f"{'SeqLen':<8} {'Dim':<5} {'Dtype':<8} | {'PyTorch (ms)':<25} | {'Triton (ms)':<25} | {'Speedup (E2E)':<10}")
    print(f"{'':<23} | {'Fwd':<10} {'Bwd':<10} | {'Fwd':<10} {'Bwd':<10} |")
    print("-" * 105)

    results = []

    for seq_len in SEQ_LENS:
        for head_dim in HEAD_DIMS:
            for dtype in DTYPES:
                shape = (BATCH_SIZE, seq_len, head_dim)
                dtype_str = "fp32" if dtype == torch.float32 else "bf16"

                # -------------------
                # 初始化数据
                # -------------------
                try:
                    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
                    k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
                    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
                    do = torch.randn(shape, device=device, dtype=dtype)
                except torch.cuda.OutOfMemoryError:
                    print(f"{seq_len:<8} {head_dim:<5} {dtype_str:<8} | OOM (Initialization phase)")
                    continue

                # -------------------
                # PyTorch Baseline
                # -------------------
                # 显存预判：如果 seq_len > 8192 且是 fp32，PyTorch 原生很可能 OOM
                # 我们这里不做硬性拦截，依靠 try-except
                
                pt_fwd, pt_bwd, pt_e2e = run_benchmark(
                    manual_attention_reference, q, k, v, IS_CAUSAL, do, "PyTorch"
                )

                # -------------------
                # Triton FlashAttn
                # -------------------
                # 清理显存
                q.grad = None; k.grad = None; v.grad = None
                torch.cuda.empty_cache()

                tri_fwd, tri_bwd, tri_e2e = run_benchmark(
                    FlashattentionWithTriton.apply, q, k, v, IS_CAUSAL, do, "Triton"
                )

                # -------------------
                # 结果记录
                # -------------------
                def fmt(x): return f"{x:.2f}" if isinstance(x, (float, int)) and x != float('inf') else "OOM"
                
                # 计算加速比
                speedup = "N/A"
                if pt_e2e != float('inf') and tri_e2e != float('inf') and tri_e2e > 0:
                    speedup = f"{pt_e2e / tri_e2e:.2f}x"

                print(f"{seq_len:<8} {head_dim:<5} {dtype_str:<8} | {fmt(pt_fwd):<10} {fmt(pt_bwd):<10} | {fmt(tri_fwd):<10} {fmt(tri_bwd):<10} | {speedup:<10}")

                results.append({
                    "SeqLen": seq_len,
                    "HeadDim": head_dim,
                    "Dtype": dtype_str,
                    "PyTorch_Fwd": pt_fwd,
                    "PyTorch_Bwd": pt_bwd,
                    "Triton_Fwd": tri_fwd,
                    "Triton_Bwd": tri_bwd,
                    "Speedup": speedup
                })

                # 删除 Tensor 释放显存
                del q, k, v, do
                torch.cuda.empty_cache()

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv("benchmark_4090d.csv", index=False)
    print("\n结果已保存至 benchmark_comparison.csv")

if __name__ == "__main__":
    main()