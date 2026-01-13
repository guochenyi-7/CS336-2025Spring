import torch
import torch.nn.functional as F
import triton
import pandas as pd
import math
import os
import gc

# -----------------------------------------------------------------------------
# 导入自定义模块
# -----------------------------------------------------------------------------
try:
    from cs336_systems.flashattention_with_triton import FlashattentionWithTriton
    from cs336_basics.attention import ScaledDotProductAttention
    HAS_LOCAL_MODULES = True
except ImportError as e:
    print(f"Warning: 无法导入自定义模块 ({e})。Triton 部分的测试将被跳过。")
    print("请确保你在项目根目录下运行此脚本。")
    HAS_LOCAL_MODULES = False
    FlashattentionWithTriton = None

# -----------------------------------------------------------------------------
# 配置参数
# -----------------------------------------------------------------------------
# 序列长度扫描范围
SEQ_LENS = [2**i for i in range(7, 15)]  # [128, 256, ..., 16384]

# Head 维度
HEAD_DIMS = [64, 128]

# 精度: 重点关注 bfloat16 (FlashAttention 的主场)
DTYPES = [torch.float32, torch.bfloat16]

BATCH_SIZE = 2      
IS_CAUSAL = True    

# -----------------------------------------------------------------------------
# PyTorch Baseline (Manual Implementation)
# -----------------------------------------------------------------------------
def manual_attention_reference(q, k, v, is_causal=False):
    scale = 1.0 / math.sqrt(q.shape[-1])
    
    # 简单的 Manual Attention
    # (B, L, D) -> (B, L, L)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    if is_causal:
        seq_len = q.shape[1]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, float('-inf'))
    
    attn_probs = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, v)
    return output

# -----------------------------------------------------------------------------
# PyTorch SDPA (Official FlashAttention)
# -----------------------------------------------------------------------------
def pytorch_sdpa_wrapper(q, k, v, is_causal):
    # 强制转换为 4D: (Batch, Seq, Dim) -> (Batch, 1, Seq, Dim)
    # PyTorch FlashAttention 后端通常需要 4D 输入 (B, H, L, D) 才能正确识别
    if q.dim() == 3:
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

    # 确保内存连续
    if not q.is_contiguous(): q = q.contiguous()
    if not k.is_contiguous(): k = k.contiguous()
    if not v.is_contiguous(): v = v.contiguous()

    # FlashAttention 仅支持 fp16 和 bf16
    use_flash = q.dtype in [torch.float16, torch.bfloat16]
    
    # 如果是 bf16/fp16，强制禁用 math (enable_math=False)
    # 这样可以验证是否真的跑了 FlashAttention。如果报错，说明环境不支持或 Shape 仍不对。
    enable_math = not use_flash 
    
    with torch.backends.cuda.sdp_kernel(enable_flash=use_flash, 
                                        enable_math=enable_math, 
                                        enable_mem_efficient=False):
        # 注意：输出也是 4D (B, 1, L, D)，如果后续需要对比，可能需要 squeeze 回去
        # 但对于 benchmark 测速来说，只要 forward 跑通即可
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

# -----------------------------------------------------------------------------
# Benchmark 核心工具函数
# -----------------------------------------------------------------------------
def run_benchmark(func, q, k, v, is_causal, grad_output, provider):
    """
    返回 (fwd_ms, bwd_ms, e2e_ms)
    """
    # 彻底清理显存
    gc.collect()
    torch.cuda.empty_cache()

    # Forward Benchmark
    try:
        # 使用 triton.testing.do_bench 自动处理 warmup 和计时
        fwd_ms = triton.testing.do_bench(lambda: func(q, k, v, is_causal), warmup=10, rep=50)
    except torch.cuda.OutOfMemoryError:
        return float('inf'), float('inf'), float('inf')
    except RuntimeError as e:
        # 如果强制禁用 math 后报错，这里会捕获到
        print(f"[{provider} FWD Error] {e}")
        return float('nan'), float('nan'), float('nan')

    # E2E (Fwd + Bwd) Benchmark
    def fwd_bwd_pass():
        # 清空梯度
        if q.grad is not None: q.grad = None
        if k.grad is not None: k.grad = None
        if v.grad is not None: v.grad = None
        
        # 前向
        o = func(q, k, v, is_causal)
        
        # 处理 SDPA 输出维度可能是 4D 的情况，对齐 grad_output
        if o.dim() == 4 and grad_output.dim() == 3:
            local_grad = grad_output.unsqueeze(1)
        else:
            local_grad = grad_output

        # 后向
        o.backward(local_grad, retain_graph=True) 
    
    try:
        gc.collect()
        torch.cuda.empty_cache()
        # grad_to_none 确保每次迭代梯度不累积
        e2e_ms = triton.testing.do_bench(fwd_bwd_pass, grad_to_none=[q, k, v], warmup=10, rep=50)
    except torch.cuda.OutOfMemoryError:
        return fwd_ms, float('inf'), float('inf')
    except Exception as e:
        print(f"[{provider} BWD Error] {e}")
        return fwd_ms, float('nan'), float('nan')

    bwd_ms = e2e_ms - fwd_ms
    return fwd_ms, bwd_ms, e2e_ms

# -----------------------------------------------------------------------------
# 主程序
# -----------------------------------------------------------------------------
def main():
    # 优化显存分配策略
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    device = torch.device("cuda")
    torch.manual_seed(0)

    print(f"Running on: {torch.cuda.get_device_name(0)}")
    print("-" * 145)
    print(f"{'SeqLen':<8} {'Dim':<5} {'Dtype':<8} | {'Manual (ms)':<17} | {'SDPA (Official)':<17} | {'Triton (Yours)':<17} | {'Speedup':<10}")
    print(f"{'':<23} | {'Fwd':<8} {'Bwd':<8} | {'Fwd':<8} {'Bwd':<8} | {'Fwd':<8} {'Bwd':<8} | {'(vs Manual)'}")
    print("-" * 145)

    results = []

    for seq_len in SEQ_LENS:
        for head_dim in HEAD_DIMS:
            for dtype in DTYPES:
                shape = (BATCH_SIZE, seq_len, head_dim)
                dtype_str = "fp32" if dtype == torch.float32 else "bf16"

                # -------------------
                # 初始化 Tensor
                # -------------------
                try:
                    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
                    k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
                    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
                    do = torch.randn(shape, device=device, dtype=dtype) # 模拟上游传来的梯度
                except torch.cuda.OutOfMemoryError:
                    print(f"{seq_len:<8} {head_dim:<5} {dtype_str:<8} | OOM at Init")
                    continue

                # ==========================================
                # PyTorch Manual (Baseline)
                # ==========================================
                pt_fwd, pt_bwd, pt_e2e = run_benchmark(
                    manual_attention_reference, q, k, v, IS_CAUSAL, do, "Manual"
                )

                # ==========================================
                # PyTorch SDPA (Official FlashAttn)
                # ==========================================
                q.grad = None; k.grad = None; v.grad = None
                sdpa_fwd, sdpa_bwd, sdpa_e2e = run_benchmark(
                    pytorch_sdpa_wrapper, q, k, v, IS_CAUSAL, do, "SDPA"
                )

                # ==========================================
                # Triton (Custom Implementation)
                # ==========================================
                tri_fwd, tri_bwd, tri_e2e = float('nan'), float('nan'), float('nan')
                
                if HAS_LOCAL_MODULES and FlashattentionWithTriton is not None:
                    q.grad = None; k.grad = None; v.grad = None
                    tri_fwd, tri_bwd, tri_e2e = run_benchmark(
                        FlashattentionWithTriton.apply, q, k, v, IS_CAUSAL, do, "Triton"
                    )

                # ==========================================
                # 记录与打印
                # ==========================================
                def fmt(x): 
                    if x == float('inf'): return "OOM"
                    if math.isnan(x): return "N/A"
                    return f"{x:.2f}"
                
                # 计算 Triton 相对于 Manual 的加速比 (基于 E2E 时间)
                speedup_str = "N/A"
                if pt_e2e != float('inf') and tri_e2e > 0 and tri_e2e != float('inf'):
                    speedup_str = f"{pt_e2e / tri_e2e:.2f}x"
                elif pt_e2e != float('inf') and sdpa_e2e > 0:
                    speedup_str = f"(SDPA {pt_e2e / sdpa_e2e:.1f}x)"

                print(f"{seq_len:<8} {head_dim:<5} {dtype_str:<8} | "
                      f"{fmt(pt_fwd):<8} {fmt(pt_bwd):<8} | "
                      f"{fmt(sdpa_fwd):<8} {fmt(sdpa_bwd):<8} | "
                      f"{fmt(tri_fwd):<8} {fmt(tri_bwd):<8} | "
                      f"{speedup_str:<10}")

                results.append({
                    "SeqLen": seq_len, "HeadDim": head_dim, "Dtype": dtype_str,
                    "Manual_Fwd": pt_fwd, "Manual_Bwd": pt_bwd,
                    "SDPA_Fwd": sdpa_fwd, "SDPA_Bwd": sdpa_bwd,
                    "Triton_Fwd": tri_fwd, "Triton_Bwd": tri_bwd,
                    "Speedup_Triton": speedup_str
                })

                # 清理本轮 Tensor
                del q, k, v, do
                torch.cuda.empty_cache()

    # 保存结果
    if results:
        df = pd.DataFrame(results)
        df.to_csv("benchmark_full_comparison.csv", index=False)
        print("\n[Done] 结果已保存至 benchmark_full_comparison.csv")
    else:
        print("\n[Error] 未生成任何结果数据。")

if __name__ == "__main__":
    main()