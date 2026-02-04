import sys
import os
import random

from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.language_identification import identify_language
# 配置路径
# 获取当前脚本所在的绝对文件夹路径
curr_dir = os.path.dirname(os.path.abspath(__file__))

# 基于当前脚本的位置拼接路径

INPUT_WET = os.path.join(curr_dir, "data", "CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz")
OUTPUT_FILE = os.path.join(curr_dir, "data", "negative_samples.txt")

# 设置目标数量（建议与正样本数量保持 1:1 或 1:2）
TARGET_COUNT = 20000

def process_negative_samples():
    count = 0
    kept = 0
    
    print(f"开始处理负样本...")
    collected_samples = []
    for record in ArchiveIterator(open(INPUT_WET, "rb"), record_types=WarcRecordType.conversion):
        if kept >= TARGET_COUNT:
            break
        try:
            text = record.reader.read().decode('utf-8')

            # 长度过滤
            if len(text.strip()) < 100:
                continue
            # 语言过滤
            lang, score = identify_language(text)
            if lang != "en" or score < 0.6:
                continue

            clean_text = text.replace('\n', ' ').replace('\r', ' ')
            collected_samples.append(clean_text)
            kept += 1
        
        except Exception as e:
            print(f"Error processing record: {e}")
            continue
        
        count += 1

        if count % 1000 == 0:
            print(f"Processed {count} records, kept {kept} high-quality examples.")

        print(f"Finished! Total processed: {count}, Total kept: {kept}")
    
     # 随机采样
    if len(collected_samples) > TARGET_COUNT:
        print(f"🎲 数据过多，随机抽取 {TARGET_COUNT} 条...")
        final_samples = random.sample(collected_samples, TARGET_COUNT)
    else:
        final_samples = collected_samples

    # 写入文件
    print(f"💾 正在写入 {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        for text in final_samples:
            out.write(f"__label__low {text}\n")
    
    print("🎉 处理完成！")

if __name__ == "__main__":
    process_negative_samples()