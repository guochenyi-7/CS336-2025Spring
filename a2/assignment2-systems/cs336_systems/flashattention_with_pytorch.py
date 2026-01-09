import torch
import math

class FlashattentionWithPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # 确保维度正确
        is_3d = False
        if Q.dim() == 3:
            is_3d = True
            Q = Q.unsqueeze(1)
            K = K.unsqueeze(1)
            V = V.unsqueeze(1)
        batch_size, n_heads, seq_len_q, head_dim = Q.shape
        seq_len_k = K.shape[-2]

        # tiling 大小
        Br = 64
        Bc = 64
        
        # 初始化张量
        O = torch.zeros_like(Q)
        L = torch.zeros((batch_size, n_heads, seq_len_q), device=Q.device, dtype=torch.float32)

        for b in range(batch_size):
            for h in range(n_heads):
                # 获取当前的Q, K, V
                q_bh = Q[b, h]
                k_bh = K[b, h]
                v_bh = V[b, h]

                # 定义块指针
                Tr = math.ceil(seq_len_q / Br)
                Tc = math.ceil(seq_len_k / Bc)

                for i in range(Tr):
                    # 获取当前块信息
                    i_start = i * Br
                    i_end = min((i + 1) * Br, seq_len_q)
                    q_tile = q_bh[i_start: i_end]
            
                    # 定义统计变量
                    # 每行当前的最大值
                    m_i = torch.full((i_end - i_start, ), float('-inf'), device=Q.device)
                    # 当前的分子
                    l_i = torch.zeros((i_end - i_start), device=Q.device)
                    # 当前的v的累积加权和
                    o_i = torch.zeros_like(q_tile)

                    for j in range(Tc):
                        # 获取当前块信息
                        j_start = j * Bc
                        j_end = min((j + 1) * Bc, seq_len_k)
                        k_tile = k_bh[j_start: j_end]
                        v_tile = v_bh[j_start: j_end]

                        # 计算 attention scores
                        s_ij = (q_tile @ k_tile.transpose(-1, -2)) / math.sqrt(head_dim)

                        # 掩码
                        if is_causal:
                            rows = torch.arange(i_start, i_end, device=Q.device).unsqueeze(1)
                            cols = torch.arange(j_start, j_end, device=Q.device).unsqueeze(0)
                            mask = cols > rows
                            s_ij.masked_fill_(mask, float('-inf'))

                        # 当前块的最大值
                        m_ij = s_ij.max(dim=-1).values
                        # 当前最大值
                        m_new = torch.maximum(m_i, m_ij)
                        # 修正因子
                        alpha = torch.exp(m_i - m_new)
                        # 当前块的未归一化概率
                        p_tilde = torch.exp(s_ij - m_new.unsqueeze(1))
                        # 修正分子
                        l_new = l_i * alpha + p_tilde.sum(dim=-1)
                        # 修正v的累积加权和
                        o_new = o_i * alpha.unsqueeze(1) + p_tilde @ v_tile
                        # 更新统计量
                        m_i = m_new
                        l_i = l_new
                        o_i = o_new
                
                    # 写入答案
                    O[b, h, i_start: i_end] = o_i / l_i.unsqueeze(1)
                    # L_i存储的就是每一行 Attention Score 做 Softmax 时的分母的对数值，这里不需要减去最大值
                    L[b, h, i_start: i_end] = m_i + torch.log(l_i)
        
        if is_3d:
            O = O.squeeze(1)
            L = L.squeeze(1)
            Q = Q.squeeze(1)
            K = K.squeeze(1)
            V = V.squeeze(1)
        # 保存变量供反向传播使用 
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O
    
    @staticmethod
    def backward(ctx, dO): 
        # 获取前向传播中保存的信息
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        
        head_dim = Q.shape[-1]
        scale = 1.0 / math.sqrt(head_dim)

        # 重计算S
        S = Q @ K.transpose(-2, -1) * scale
        # mask
        if is_causal:
            seq_len_q = Q.shape[-2]
            seq_len_k = K.shape[-2]
            rows = torch.arange(seq_len_q, device=Q.device).unsqueeze(1)
            cols = torch.arange(seq_len_k, device=K.device).unsqueeze(0)
            mask = cols > rows
            S = S.masked_fill(mask, float('-inf'))
        # 计算P
        P = torch.exp(S - L.unsqueeze(-1))
        # 计算dV = P_t @ dO
        dV = P.transpose(-2, -1) @ dO
        # 计算dP = dO @ V_t
        dP = dO @ V.transpose(-2, -1)
        # 计算D = rowsum(dO 点乘 O)
        D = torch.sum(dO * O, dim=-1, keepdim=True)
        # 计算dS = P 点乘(dP - D)
        dS = P * (dP - D)
        # 计算dQ = scale * dS @ K
        dQ = dS @ K * scale
        # 计算dK = scale *。dS_t @ Q
        dK = dS.transpose(-2, -1) @ Q * scale
        return dQ, dK, dV, None


def flash_attention_reference(q, k, v, is_causal=False):
    return FlashattentionWithPytorch.apply(q, k, v, is_causal)
