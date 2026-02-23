import json

correct_both = 0  # 格式1，答案1
format_only = 0   # 格式1，答案0
failed_both = 0   # 格式0，答案0

with open("results.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        res = json.loads(line)
        f_rew = res["scores"].get("format_reward", 0)
        a_rew = res["scores"].get("answer_reward", 0)
        
        if f_rew == 1 and a_rew == 1:
            correct_both += 1
        elif f_rew == 1 and a_rew == 0:
            format_only += 1
        else:
            failed_both += 1

print(f"统计结果:")
print(f"(1) 正确 (格式1, 答案1): {correct_both}")
print(f"(2) 仅格式正确 (格式1, 答案0): {format_only}")
print(f"(3) 全错 (格式0, 答案0): {failed_both}")