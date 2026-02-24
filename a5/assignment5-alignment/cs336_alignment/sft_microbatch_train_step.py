import torch

from cs336_alignment.masked_normalize import masked_normalize

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    在一个微批次上执行前向和反向传播。
    """
    # 计算负对数似然
    nll = -policy_log_probs

    # 应用掩码计算归一化的损失
    per_example_loss = masked_normalize(
        tensor=nll,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=1,
    )

    # 在batch维度上取平均
    loss = per_example_loss.mean()

    # 梯度累加缩放
    scaled_loss = loss / gradient_accumulation_steps

    # 反向传播
    scaled_loss.backward()

    # 准备元数据
    metadata = {
        "unscaled_loss": loss.detach(),
    }

    return scaled_loss, metadata
