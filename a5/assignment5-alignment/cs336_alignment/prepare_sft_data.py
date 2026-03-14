import json
from pathlib import Path

# R1-Zero 提示词模板
PROMPT_TEMPLATE = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer.
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>\n"""

def process_gsm8k_to_sft_format(input_file: str, output_file: str):
    processed_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            question = item.get("question", "")
            raw_answer = item.get("answer", "")
            
            # GSM8K 的答案格式是用 #### 分隔推理过程和最终结果
            if "####" in raw_answer:
                parts = raw_answer.split("####")
                reasoning = parts[0].strip()
                final_answer = parts[1].strip()
            else:
                # 如果遇到没有 #### 的异常数据，直接跳过
                continue
            
            # 1. 组装 prompt：将问题填入 R1-Zero 模板
            prompt = PROMPT_TEMPLATE.format(question=question)
            
            # 2. 组装 response：接着 prompt 最后的 <think>\n 开始写推理过程，然后闭合标签并加上 answer
            response = f"{reasoning}\n</think> <answer> {final_answer} </answer>"

            processed_data.append({
                "prompt": prompt,
                "response": response
            })
            
    # 将处理好的数据写入新的 jsonl 文件
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for data in processed_data:
            out_f.write(json.dumps(data, ensure_ascii=False) + '\n')
            
    print(f"数据处理完成！")
    print(f"共成功转换了 {len(processed_data)} 条数据。")
    print(f"已保存至: {output_file}")

if __name__ == "__main__":
    # 根据你的项目目录结构设置路径
    project_root = Path(__file__).parent.parent  # 如果脚本在 cs336_alignment 目录下
    
    input_path = project_root / "data" / "gsm8k" / "train.jsonl"
    output_path = project_root / "data" / "gsm8k" / "sft_formatted.jsonl"
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    process_gsm8k_to_sft_format(str(input_path), str(output_path))