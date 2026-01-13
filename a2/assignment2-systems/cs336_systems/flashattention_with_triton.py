from typing import Any
import torch
import triton
import math
import triton.language as tl

tile_size = 32

@triton.jit
def flashattention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    Q_stride_batch, Q_stride_row, Q_stride_dim,
    K_stride_batch, K_stride_row, K_stride_dim,
    V_stride_batch, V_stride_row, V_stride_dim,
    O_stride_batch, O_stride_row, O_stride_dim,
    L_stride_batch, L_stride_row,
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
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_idx * O_stride_batch,
        shape=(seq_len_q, head_dim),
        strides=(O_stride_row, O_stride_dim),
        offsets=(row_tile_idx * Q_tile_size, 0),
        block_shape=(Q_tile_size, head_dim),
        order=(1, 0)
    )
    # 构造l指针
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_idx * L_stride_batch,
        shape=(seq_len_q, ),
        strides=(L_stride_row, ),
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
        offs_k = (j * K_tile_size + offs_k_base)[None, :]
        mask_boundary = offs_k < seq_len_k
        if is_causal:
            mask_causal = offs_k <= offs_q
            mask = mask_causal & mask_boundary
        else:
            mask = mask_boundary
         # 应用掩码
        s_ij = tl.where(mask, s_ij, -1.0e6)

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
    O_final = o_i / l_i[:, None]
    tl.store(O_block_ptr, O_final.to(O_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, m_i + tl.log(l_i), boundary_check=(0,))


@triton.jit
def flashattention_bwd_kv_kernel(
    Q_ptr, K_ptr, V_ptr, L_ptr, D_ptr, dO_ptr, dK_ptr, dV_ptr,
    Q_stride_batch, Q_stride_row, Q_stride_dim,
    K_stride_batch, K_stride_row, K_stride_dim,
    V_stride_batch, V_stride_row, V_stride_dim,
    dO_stride_batch, dO_stride_row, dO_stride_dim,
    dK_stride_batch, dK_stride_row, dK_stride_dim,
    dV_stride_batch, dV_stride_row, dV_stride_dim,
    L_stride_batch, L_stride_row,
    D_stride_batch, D_stride_row,
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
        offsets=(0, 0),
        block_shape=(Q_tile_size, head_dim),
        order=(1, 0)
    )

    # 构造K指针
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_idx * K_stride_batch,
        shape=(seq_len_k, head_dim),
        strides=(K_stride_row, K_stride_dim),
        offsets=(row_tile_idx * K_tile_size, 0),
        block_shape=(K_tile_size, head_dim),
        order=(1, 0)
    )

    # 构造V指针
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_idx * V_stride_batch,
        shape=(seq_len_k, head_dim),
        strides=(V_stride_row, V_stride_dim),
        offsets=(row_tile_idx * K_tile_size, 0),
        block_shape=(K_tile_size, head_dim),
        order=(1, 0)
    )
    # 构造L指针
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_idx * L_stride_batch,
        shape=(seq_len_q, ),
        strides=(L_stride_row, ),
        offsets=(0, ),
        block_shape=(Q_tile_size, ),
        order=(0, )        
    )
    # 构造D指针
    D_block_ptr = tl.make_block_ptr(
        base=D_ptr + batch_idx * D_stride_batch,
        shape=(seq_len_q, ),
        strides=(D_stride_row, ),
        offsets=(0, ),
        block_shape=(Q_tile_size, ),
        order=(0, )        
    )
    # 构造dK指针
    dK_block_ptr = tl.make_block_ptr(
        base=dK_ptr + batch_idx * dK_stride_batch,
        shape=(seq_len_k, head_dim),
        strides=(dK_stride_row, dK_stride_dim),
        offsets=(row_tile_idx * K_tile_size, 0),
        block_shape=(K_tile_size, head_dim),
        order=(1, 0)
    )
    # 构造dV指针
    dV_block_ptr = tl.make_block_ptr(
        base=dV_ptr + batch_idx * dV_stride_batch,
        shape=(seq_len_k, head_dim),
        strides=(dV_stride_row, dV_stride_dim),
        offsets=(row_tile_idx * K_tile_size, 0),
        block_shape=(K_tile_size, head_dim),
        order=(1, 0)
    )
    # 构造dO指针
    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr + batch_idx * dO_stride_batch,
        shape=(seq_len_q, head_dim),
        strides=(dO_stride_row, dO_stride_dim),
        offsets=(0, 0),
        block_shape=(Q_tile_size, head_dim),
        order=(1, 0)
    )
    
    # 加载当前K, V块
    K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # 初始化累加器
    dK_tile = tl.zeros((K_tile_size, head_dim), dtype=tl.float32)
    dV_tile = tl.zeros((K_tile_size, head_dim), dtype=tl.float32)

    # 掩码准备
    offs_q_base = tl.arange(0, Q_tile_size)
    offs_k_base = tl.arange(0, K_tile_size)
    offs_k = (row_tile_idx * K_tile_size + offs_k_base)[None, :]


    for i in range(tl.cdiv(seq_len_q, Q_tile_size)):
        # 加载数据
        Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        dO_tile = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        D_tile = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
        L_tile = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        
        s_ij = tl.dot(Q_tile, tl.trans(K_tile)) * scale
        # 掩码
        offs_q = (i * Q_tile_size + offs_q_base)[:, None]
        mask_boundary = offs_q < seq_len_q
        if is_causal:
            mask_causal = offs_k <= offs_q
            mask =  mask_causal & mask_boundary
        else:
            mask = mask_boundary
        # 应用掩码
        s_ij = tl.where(mask, s_ij, -1.0e6)

        p_ij = tl.exp(s_ij - L_tile[:, None])
        
        # 计算V的部分的部分梯度
        dV_tile += tl.dot(tl.trans(p_ij.to(Q_tile.dtype)), dO_tile)

        # 计算K的部分的部分梯度
        dp_ij = tl.dot(dO_tile, tl.trans(V_tile))
        dS_ij = p_ij * (dp_ij - D_tile[:, None])
        dK_tile += tl.dot(tl.trans(dS_ij.to(Q_tile.dtype)), Q_tile)

        # 指针移动
        Q_block_ptr = tl.advance(Q_block_ptr, (Q_tile_size, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_tile_size, 0))
        D_block_ptr = tl.advance(D_block_ptr, (Q_tile_size,))
        L_block_ptr = tl.advance(L_block_ptr, (Q_tile_size,))
    
    dK_tile = dK_tile * scale
    # 写回
    tl.store(dK_block_ptr, dK_tile.to(dK_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dV_tile.to(dV_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def flashattention_bwd_q_kernel(
    Q_ptr, K_ptr, V_ptr, L_ptr, D_ptr, dO_ptr, dQ_ptr,
    Q_stride_batch, Q_stride_row, Q_stride_dim,
    K_stride_batch, K_stride_row, K_stride_dim,
    V_stride_batch, V_stride_row, V_stride_dim,
    dO_stride_batch, dO_stride_row, dO_stride_dim,
    dQ_stride_batch, dQ_stride_row, dQ_stride_dim,
    L_stride_batch, L_stride_row,
    D_stride_batch, D_stride_row,
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
    # 构造L指针
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_idx * L_stride_batch,
        shape=(seq_len_q, ),
        strides=(L_stride_row, ),
        offsets=(row_tile_idx * Q_tile_size, ),
        block_shape=(Q_tile_size, ),
        order=(0, )        
    )
    # 构造D指针
    D_block_ptr = tl.make_block_ptr(
        base=D_ptr + batch_idx * D_stride_batch,
        shape=(seq_len_q, ),
        strides=(D_stride_row, ),
        offsets=(row_tile_idx * Q_tile_size, ),
        block_shape=(Q_tile_size, ),
        order=(0, )        
    )
    # 构造dO指针
    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr + batch_idx * dO_stride_batch,
        shape=(seq_len_q, head_dim),
        strides=(dO_stride_row, dO_stride_dim),
        offsets=(row_tile_idx * Q_tile_size, 0),
        block_shape=(Q_tile_size, head_dim),
        order=(1, 0)
    )
    # 构造dQ指针
    dQ_block_ptr = tl.make_block_ptr(
        base=dQ_ptr + batch_idx * dQ_stride_batch,
        shape=(seq_len_q, head_dim),
        strides=(dQ_stride_row, dQ_stride_dim),
        offsets=(row_tile_idx * Q_tile_size, 0),
        block_shape=(Q_tile_size, head_dim),
        order=(1, 0)
    )
    
    # 加载当前块数据
    Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dO_tile = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
    D_tile = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")
    L_tile = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")

    # 初始化累加器
    dQ_tile = tl.zeros((Q_tile_size, head_dim), dtype=tl.float32)

    # 掩码准备
    offs_q_base = tl.arange(0, Q_tile_size)
    offs_k_base = tl.arange(0, K_tile_size)
    offs_q = (row_tile_idx * Q_tile_size + offs_q_base)[:, None]


    for j in range(tl.cdiv(seq_len_k, K_tile_size)):
        # 加载数据
        K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        s_ij = tl.dot(Q_tile, tl.trans(K_tile)) * scale

        # 掩码
        offs_k = (j * K_tile_size + offs_k_base)[None, :]
        mask_boundary = offs_k < seq_len_k
        if is_causal:
            mask_causal = offs_k <= offs_q
            mask =  mask_causal & mask_boundary
        else:
            mask = mask_boundary

        # 应用掩码
        s_ij = tl.where(mask, s_ij, -1.0e6)

        # 计算Q的部分的部分梯度c
        p_ij = tl.exp(s_ij - L_tile[:, None])
        dp_ij = tl.dot(dO_tile, tl.trans(V_tile))
        dS_ij = p_ij * (dp_ij - D_tile[:, None])
        dQ_tile += tl.dot(dS_ij.to(Q_tile.dtype), K_tile)

        # 指针移动
        K_block_ptr = tl.advance(K_block_ptr, (K_tile_size, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_tile_size, 0))
    
    dQ_tile = dQ_tile * scale
    # 写回
    tl.store(dQ_block_ptr, dQ_tile.to(dQ_ptr.dtype.element_ty), boundary_check=(0, 1))
    


class FlashattentionWithTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # 确保连续
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        # 获取维度信息
        batch_size, seq_len_q, head_dim = Q.shape
        seq_len_k = K.shape[-2]

        # 设置分块大小
        Q_tile_size = tile_size
        K_tile_size = tile_size

        # 初始化输出
        O = torch.zeros_like(Q)
        L = torch.empty((batch_size, seq_len_q), device=Q.device, dtype=torch.float32)
        
        scale = 1.0 / math.sqrt(head_dim)
        # 启动kernel
        grid = (triton.cdiv(seq_len_q, Q_tile_size), batch_size)
        flashattention_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            seq_len_q, seq_len_k,
            scale, 
            head_dim=head_dim,
            Q_tile_size=Q_tile_size,
            K_tile_size=K_tile_size,
            is_causal=is_causal,
        )
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, dO):
        # 确保连续
        dO = dO.contiguous()
        # 获取前向传播中数据
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        batch_size, seq_len_q, head_dim = Q.shape
        seq_len_k = K.shape[-2]
        scale = 1.0 / math.sqrt(head_dim)

        # 设置分块大小
        Q_tile_size = tile_size
        K_tile_size = tile_size

        # 初始化输出
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        # 计算中间变量D
        D = (dO * O).sum(dim=-1, keepdim=True)

        # 启动计算dK, dV的内核
        grid = (triton.cdiv(seq_len_k, K_tile_size), batch_size)
        flashattention_bwd_kv_kernel[grid](
            Q, K, V, L, D, dO, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            seq_len_q, seq_len_k,
            scale, 
            head_dim=head_dim,
            Q_tile_size=Q_tile_size,
            K_tile_size=K_tile_size,
            is_causal=is_causal,
        )

        #  启动计算dQ的内核
        grid = (triton.cdiv(seq_len_q, Q_tile_size), batch_size)
        flashattention_bwd_q_kernel[grid](
            Q, K, V, L, D, dO, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            seq_len_q, seq_len_k,
            scale, 
            head_dim=head_dim,
            Q_tile_size=Q_tile_size,
            K_tile_size=K_tile_size,
            is_causal=is_causal,
        )

        return dQ, dK, dV, None
