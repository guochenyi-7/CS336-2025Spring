import gzip
import random
import os
# 配置路径
# 获取当前脚本所在的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 使用 os.path.join 拼接路径
INPUT_FILE = os.path.join(BASE_DIR, "data", "enwiki-20240420-extracted_urls.txt.gz")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "subsampled_positive_urls.txt")
SAMPLE_SIZE = 20000  # 采样数量

print(f"正在读取 {INPUT_FILE}...")
with gzip.open(INPUT_FILE, "rt", encoding="utf-8") as f:
    all_urls = f.read().splitlines()

print(f"总 URL 数: {len(all_urls)}")
sampled_urls = random.sample(all_urls, SAMPLE_SIZE)

print(f"正在写入 {SAMPLE_SIZE} 条 URL 到 {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.writelines(url + '\n' for url in sampled_urls)
