import json
import os
import random

# 配置路径
INPUT_DIR = "data/extracted_wiki"  # WikiExtractor 输出的文件夹
OUTPUT_FILE = "data/positive_samples.txt"
SAMPLE_SIZE = 9883 

def process_wiki_data():
    # --- 调试信息 ---
    print(f"当前工作目录: {os.getcwd()}")
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 错误: 找不到目录 '{INPUT_DIR}'")
        return
    else:
        print(f"✅ 找到目录 '{INPUT_DIR}'，正在扫描文件...")

    # --- 核心修改：使用 os.walk 递归查找所有文件 ---
    all_files = []
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            # 过滤掉可能的隐藏文件或系统文件
            if not file.startswith('.'):
                full_path = os.path.join(root, file)
                all_files.append(full_path)

    print(f"📂 找到 {len(all_files)} 个数据文件，开始处理...")
    
    if len(all_files) == 0:
        print("⚠️ 警告：目录是空的，请检查 extracted_wiki 文件夹里有没有内容！")
        return

    collected_samples = []

    # 遍历所有找到的文件
    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = data.get('text', '')
                        title = data.get('title', '')
                        
                        # 拼接标题和正文
                        full_text = f"{title}. {text}"
                        
                        # 清洗换行符 (Critical)
                        clean_text = full_text.replace('\n', ' ').replace('\r', ' ')
                        # 压缩多余空格
                        clean_text = ' '.join(clean_text.split())

                        # 长度过滤
                        if len(clean_text) > 100: # 稍微提高一点门槛
                            collected_samples.append(clean_text)

                    except json.JSONDecodeError:
                        continue
        except IsADirectoryError:
            continue
    
    print(f"✅ 总共提取了 {len(collected_samples)} 条原始文章。")

    # 随机采样
    if len(collected_samples) > SAMPLE_SIZE:
        print(f"🎲 数据过多，随机抽取 {SAMPLE_SIZE} 条...")
        final_samples = random.sample(collected_samples, SAMPLE_SIZE)
    else:
        final_samples = collected_samples

    # 写入文件
    print(f"💾 正在写入 {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        for text in final_samples:
            out.write(f"__label__hq {text}\n")
    
    print("🎉 处理完成！")

if __name__ == "__main__":
    process_wiki_data()