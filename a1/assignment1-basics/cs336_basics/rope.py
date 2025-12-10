import torch
import torch.nn as nn

from einops import rearrange
from torch import Tensor
from jaxtyping import Float, Int

class RoPE(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 计算频率倒数
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        #生成索引
        t = torch.arange(max_seq_len, device=device).float()
        # 计算角度矩阵
        # [max_seq_len, d_k // 2]
        freqs = torch.outer(t, inv_freq)
        # 缓存
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)
    
    def forward(
            self, 
            x: Float[Tensor, "... seq_len d_k"],
            token_positions: Int[Tensor, "... seq_len"],
        ) -> torch.Tensor:
        self.cos_cached: torch.Tensor 
        self.sin_cached: torch.Tensor
        # 获取对应的cos和sin
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        if x.ndim == 4:
            # 训练模式：插入 Head 维度以便广播
            # cos: [Batch, Seq, Half_Dim] -> [Batch, 1, Seq, Half_Dim]
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        # 重排x,两两一对
        x_pairs = rearrange(x, "... (d two) -> ... d two", two=2)
        # 取出奇偶
        x_r = x_pairs[..., 0]
        x_i = x_pairs[..., 1]
        # 4. 应用旋转公式
        #   [ cos -sin ] [ x_r ]   [ x_r*cos - x_i*sin ]
        #   [ sin  cos ] [ x_i ] = [ x_r*sin + x_i*cos ]
        # cos 和 sin 会自动广播到匹配 x_r 和 x_i 的形状
        x_out_r = x_r * cos - x_i * sin
        x_out_i = x_r * sin + x_i * cos
        # 恢复原状
        x_out = rearrange([x_out_r, x_out_i], "two ... d -> ... (d two)")
        return x_out
        