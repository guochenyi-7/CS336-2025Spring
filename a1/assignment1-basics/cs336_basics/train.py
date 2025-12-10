import torch
import wandb
import time
import argparse
import os

import numpy as np

from pathlib import Path
from .transformerLM import TransformerLM
from .optimizer import MyAdamW, lr_cosine_schedule, gradient_clipping
from .utils import load_checkpoint, get_batch, cross_entropy_loss, save_checkpoint


def train(args):
    # 设置设备
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Using device: {device}")
    torch.manual_seed(args.seed)

    # 初始化wandb
    if args.use_wandb:
        wandb.init(project="cs336-a1", config=args)

    # 数据加载
    train_data = np.memmap(args.train_data_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_data_path, dtype=np.uint16, mode="r")

    # 类型转换
    pt_dtype = torch.float32
    if args.dtype == "float16":
        pt_dtype = torch.float16
    elif args.dtype == "bfloat16":
        pt_dtype = torch.bfloat16
    # 初始化模型
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_size=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        dtype=pt_dtype,
    )
    model.to(device)
    # 初始化优化器
    optimizer = MyAdamW(
        params=model.parameters(),
        lr=args.learning_rate,
        betas=args.betas,
        weight_decay=args.weight_decay
    )

    # 训练循环
    iter_num = 0
    # 如果有恢复检查点需求
    if args.resume_from:
        iter_num = load_checkpoint(args.resume_from, model, optimizer)

    print("开始训练")
    t0 = time.time()
    while iter_num < args.max_iters:
        # 获取batch
        X, Y = get_batch(train_data, args.batch_size, args.context_length, device)
        # 学习率调度
        lr = lr_cosine_schedule(
            it=iter_num,
            max_learning_rate=args.max_learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters
        )
        for param_group in optimizer.param_groups:
            param_group["lr"]=lr
        #  前向传播
        optimizer.zero_grad()
        logits = model(X)
        loss = cross_entropy_loss(logits, Y)
        
        # 反向传播
        loss.backward()
        # 梯度裁剪
        gradient_clipping(model.parameters(), args.max_l2_norm)
        # 优化
        optimizer.step()

        #  日志记录
        if iter_num % args.log_interval == 0:
            t1 = time.time()
            dt = (t1 - t0) / args.log_interval
            t0 = t1
            tokens_per_sec = (args.batch_size * args.context_length) / dt
            print(f"Iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.2f}ms, tok/s {tokens_per_sec:.2f}")
            if args.use_wandb:
                wandb.log({"train/loss": loss.item(), "iter": iter_num, "lr": lr})
        
        # 验证集评估
        if iter_num % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                # 为了节省时间，我们只从验证集里随机取 20 个 batch 来估算 Loss
                val_losses = []
                for _ in range(20):
                    X_val, Y_val = get_batch(val_data, args.batch_size, args.context_length, device)
                    logits_val = model(X_val)
                    loss_val = cross_entropy_loss(logits_val, Y_val)
                    val_losses.append(loss_val.item())
                
                avg_val_loss = sum(val_losses) / len(val_losses)
                print(f"Validation Loss: {avg_val_loss:.4f}")
                
                if args.use_wandb:
                    wandb.log({"val/loss": avg_val_loss, "iter": iter_num})
            model.train()
        
        if iter_num > 0 and iter_num % args.save_interval == 0:
            if args.out_dir:
                os.makedirs(args.out_dir, exist_ok=True)
            save_path = Path(args.out_dir) / f"ckpt_{iter_num}.pt"
            save_checkpoint(model, optimizer, iter_num, save_path)
        
        iter_num += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # -------------------------------------------------------------------------
    # 模型架构参数 (Model Architecture)
    # -------------------------------------------------------------------------
    parser.add_argument('--vocab_size', type=int, default=10000, help="词汇表大小")
    parser.add_argument('--context_length', type=int, default=256, help="上下文长度")
    parser.add_argument('--d_model', type=int, default=512, help="嵌入维度")
    parser.add_argument('--num_layers', type=int, default=4, help="Transformer 层数")
    parser.add_argument('--num_heads', type=int, default=16, help="注意力头数")
    parser.add_argument('--d_ff', type=int, default=1344, help="前馈网络维度")
    parser.add_argument('--rope_theta', type=float, default=10000.0, help="RoPE 的 theta 参数")
    parser.add_argument('--dtype', type=str, default="float32", choices=["float32", "float16", "bfloat16"], help="模型权重的数据类型")

    # -------------------------------------------------------------------------
    # 训练与优化器参数 (Training & Optimizer)
    # -------------------------------------------------------------------------
    parser.add_argument('--batch_size', type=int, default=32, help="批次大小")
    # 学习率相关
    parser.add_argument('--learning_rate', type=float, default=5e-4, help="初始/基础学习率 (AdamW 的 lr)")
    parser.add_argument('--max_learning_rate', type=float, default=5e-4, help="Cosine 调度器的最大学习率")
    parser.add_argument('--min_learning_rate', type=float, default=5e-5, help="Cosine 调度器的最大学习率 (通常为 max 的 10%)")
    # 优化器细节
    parser.add_argument('--weight_decay', type=float, default=0.1, help="权重衰减 (Typical for LLMs)")
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.95), help="AdamW betas")
    parser.add_argument('--max_l2_norm', type=float, default=1.0, help="梯度裁剪的最大 L2 范数")

    # -------------------------------------------------------------------------
    # 训练循环控制 (Loop Control)
    # -------------------------------------------------------------------------
    parser.add_argument('--max_iters', type=int, default=100, help="总训练步数")
    parser.add_argument('--warmup_iters', type=int, default=200, help="预热步数 (通常为总步数的 1%-5%)")
    parser.add_argument('--cosine_cycle_iters', type=int, default=5000, help="Cosine 调度周期的长度 (通常等于 max_iters)")
    
    # -------------------------------------------------------------------------
    # 系统与日志 (System & Logging)
    # -------------------------------------------------------------------------
    parser.add_argument('--seed', type=int, default=42, help="随机种子，保证可复现性")
    parser.add_argument('--device', type=str, default="auto", help="训练设备: 'cuda', 'mps', 'cpu' 或 'auto'")
    
    # 路径与检查点
    # parser.add_argument('--train_data_path', type=str, required=True, help="训练数据 .bin 文件路径")
    # parser.add_argument('--val_data_path', type=str, required=True, help="验证数据 .bin 文件路径")
    parser.add_argument('--train_data_path', type=str, default="data/tinystories_train.bin", help="训练数据路径")
    parser.add_argument('--val_data_path', type=str, default="data/tinystories_val.bin", help="验证数据路径")
    parser.add_argument('--out_dir', type=str, default="out", help="检查点输出目录")
    parser.add_argument('--resume_from', type=str, default=None, help="从指定检查点路径恢复训练")

    # 日志频率
    parser.add_argument('--log_interval', type=int, default=10, help="每多少步打印一次日志")
    parser.add_argument('--eval_interval', type=int, default=200, help="每多少步评估一次验证集")
    parser.add_argument('--save_interval', type=int, default=1000, help="每多少步保存一次检查点")
    parser.add_argument('--use_wandb', action='store_true', help="是否使用 Weights & Biases 记录实验")
    
    args = parser.parse_args()
    train(args)