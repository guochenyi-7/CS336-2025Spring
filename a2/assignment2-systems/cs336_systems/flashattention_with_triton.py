from typing import Any
import torch
import triton
import math
import triton.language as tl

@triton.jit
def flashattention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, output_ptr, l_ptr,
    Q_stride_batch, Q_stride_row, Q_stride_dim,
    K_stride_batch, K_stride_row, K_stride_dim,
    V_stride_batch, V_stride_row, V_stride_dim,
    O_stride_batch, O_stride_row, O_stride_dim,
    l_stride_batch, l_stride_row,
    seq_len_q, seq_len_k, 
    scale, 
    head_dim: tl.constexpr,
    Q_tile_size: tl.constexpr,
    K_tile_size: tl.constexpr,
    is_causal: tl.constexpr,
):
    # 获取当前块的信息
    row_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    # 构造Q指针
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_idx * Q_stride_batch,
        shape=(seq_len_q, head_dim),
        strides=(Q_stride_row, Q_stride_dim),
        offsets=(row_tile_idx * Q_tile_size, 0),
        block_shape=(Q_tile_size, head_dim),
        order=(1, 0)
    )

    # 构造K指针
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_idx * K_stride_batch,
        shape=(seq_len_k, head_dim),
        strides=(K_stride_row, K_stride_dim),
        offsets=(0, 0),
        block_shape=(K_tile_size, head_dim),
        order=(1, 0)
    )

    # 构造V指针
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_idx * V_stride_batch,
        shape=(seq_len_k, head_dim),
        strides=(V_stride_row, V_stride_dim),
        offsets=(0, 0),
        block_shape=(K_tile_size, head_dim),
        order=(1, 0)
    )
    # 构造output指针
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr + batch_idx * O_stride_batch,
        shape=(seq_len_q, head_dim),
        strides=(O_stride_row, O_stride_dim),
        offsets=(row_tile_idx * Q_tile_size, 0),
        block_shape=(Q_tile_size, head_dim),
        order=(1, 0)
    )
    # 构造l指针
    l_block_ptr = tl.make_block_ptr(
        base=l_ptr + batch_idx * l_stride_batch,
        shape=(seq_len_q, ),
        strides=(l_stride_row, ),
        offsets=(row_tile_idx * Q_tile_size, ),
        block_shape=(Q_tile_size, ),
        order=(0, )        
    )

    # 定义统计变量
    # 每行当前的最大值
    m_i = tl.full((Q_tile_size, ), float('-inf'), dtype=tl.float32)
    # 当前的分子
    l_i = tl.zeros((Q_tile_size, ), dtype=tl.float32)
    # 当前的v的累积加权和
    o_i = tl.zeros((Q_tile_size, head_dim), dtype=tl.float32)

    Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # 掩码准备
    offs_q_base = tl.arange(0, Q_tile_size)
    offs_k_base = tl.arange(0, K_tile_size)

    offs_q = (row_tile_idx * Q_tile_size + offs_q_base)[:, None]
    for j in range(tl.cdiv(seq_len_k, K_tile_size)):
        # 加载数据
        K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # 计算 attention scores
        s_ij = tl.dot(Q_tile, tl.trans(K_tile), acc=None, allow_tf32=True) * scale

        # 掩码
        if is_causal:
            offs_k = (j * K_tile_size + offs_k_base)[None, :]

            # 生成掩码 
            mask = offs_k > offs_q

            # 应用掩码
            s_ij = tl.where(mask, -1.0e6, s_ij)

        # 当前块的最大值
        m_ij = tl.max(s_ij, axis=1)
        # 获取当前最大值
        m_new = tl.maximum(m_ij, m_i)
        # 修正因子
        alpha = tl.exp(m_i - m_new)
        # 当前块为归一化概率
        p_tilde = tl.exp(s_ij - m_new[:, None])
        # 修正分子
        l_new = l_i * alpha + p_tilde.sum(axis=-1)
        # 修正v的累积加权和
        o_new = o_i * alpha[:, None] + tl.dot(p_tilde.to(V_tile.dtype), V_tile)
        # 更新统计量
        m_i = m_new
        l_i = l_new
        o_i = o_new

        # 移动指针
        K_block_ptr = tl.advance(K_block_ptr, (K_tile_size, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_tile_size, 0))
    
    # 写入答案
    output_final = o_i / l_i[:, None]
    tl.store(output_block_ptr, output_final.to(output_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(l_block_ptr, m_i + tl.log(l_i), boundary_check=(0,))


class FlashattentionWithTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # 获取维度信息
        batch_size, seq_len_q, head_dim = Q.shape
        seq_len_k = K.shape[-2]

        # 设置分块大小
        Q_tile_size = 64
        K_tile_size = 64

        # 初始化输出
        O = torch.zeros_like(Q)
        l = torch.empty((batch_size, seq_len_q), device=Q.device, dtype=torch.float32)
        
        scale = 1.0 / math.sqrt(head_dim)
        # 启动kernel
        grid = (triton.cdiv(seq_len_q, Q_tile_size), batch_size)
        flashattention_fwd_kernel[grid](
            Q, K, V, O, l,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            l.stride(0), l.stride(1),
            seq_len_q, seq_len_k,
            scale, 
            head_dim=head_dim,
            Q_tile_size=Q_tile_size,
            K_tile_size=K_tile_size,
            is_causal=is_causal,
        )
        ctx.save_for_backward(Q, K, V, O, l)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError("Backward pass not implemented yet")
