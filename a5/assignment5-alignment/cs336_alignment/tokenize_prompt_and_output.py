import torch

from transformers import PreTrainedTokenizer

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
):
    """
    将提示词和输出字符串分别进行分词，拼接它们，并构建一个掩码。
    掩码在响应token 处为 1 在提示词或填充处为 0 
    
    Args:
        prompt_strs: 包含提示词字符串的列表。
        output_strs: 包含输出字符串的列表。
        tokenizer: 用于分词的预训练 Tokenizer。
        
    Returns:
        包含 input_ids, labels, 和 response_mask 的字典。
    """
    batch_full_ids = []
    batch_masks = []
    max_len = 0

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        output_ids = tokenizer(output, add_special_tokens=False)["input_ids"]

        full_ids = prompt_ids + output_ids

        max_len = max(max_len, len(full_ids))

        mask = [0] * len(prompt_ids) + [1] * len(output_ids)

        batch_full_ids.append(full_ids)
        batch_masks.append(mask)

    # 获取pad_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    padded_ids = []
    padded_masks = []

    for ids, mask in zip(batch_full_ids, batch_masks):
        pad_len = max_len - len(ids)
        padded_ids.append(ids + [pad_token_id] * pad_len)
        padded_masks.append(mask + [0] * pad_len)

    full_tensor = torch.tensor(padded_ids, dtype=torch.long)
    mask_tensor = torch.tensor(padded_masks, dtype=torch.long)

    # 错位
    input_ids = full_tensor[:, :-1]
    labels = full_tensor[:, 1:]
    response_mask = mask_tensor[:, 1:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }
