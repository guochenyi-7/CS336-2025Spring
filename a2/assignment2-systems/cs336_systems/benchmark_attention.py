import torch

from cs336_basics.attention import ScaledDotProductAttention

def benchmark_attention():
    attention = ScaledDotProductAttention()

    batch_size = 8
    d_model_list = [16, 32, 64, 128]
    seq_len_list = [256, 1024, 4096, 8192, 16384]
    print(f"{'d_model':<10} {'seq_len':<10} {'Fwd Time':<15} {'Bwd Time':<15} {'Mem (MB)':<15}")
    print("-" * 65)
    for d_k in d_model_list:
        for seq_len in seq_len_list:
            try:
                Q = torch.randn(batch_size, seq_len, d_k, device='cuda', dtype=torch.float32, requires_grad=True)
                K = torch.randn(batch_size, seq_len, d_k, device='cuda', dtype=torch.float32, requires_grad=True)
                V = torch.randn(batch_size, seq_len, d_k, device='cuda', dtype=torch.float32, requires_grad=True)

                # 预热
                for _ in range(10):
                    out = attention(Q, K, V)
                    out.sum().backward()
                    Q.grad = None; K.grad = None; V.grad = None
                torch.cuda.synchronize()
                # 测量前向传播
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                
                start_evt.record()
                out = None
                for _ in range(100):
                    out = attention(Q, K, V)
                end_evt.record()
                torch.cuda.synchronize()
                fwd_time = start_evt.elapsed_time(end_evt) / 100
                del out 
                torch.cuda.empty_cache()
                # 测量峰值显存
                torch.cuda.reset_peak_memory_stats()
                out = attention(Q, K, V)
                mem_bytes = torch.cuda.max_memory_allocated()
                mem_mb = mem_bytes / (1024 ** 2)

                out.sum().backward() 
                Q.grad = None; K.grad = None; V.grad = None
                torch.cuda.synchronize()
                # 测量反向传播
                bwd_times = []
                for _ in range(100):
                    out = attention(Q, K, V)
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
                print(f"{d_k:<10} {seq_len:<10} {fwd_time:<15.3f} {avg_bwd_time:<15.3f} {mem_mb:<15.2f}")
            except torch.cuda.OutOfMemoryError:
                print("OOM error")
                torch.cuda.empty_cache()

                
def main():
    benchmark_attention()

if __name__ == "__main__":
    main()