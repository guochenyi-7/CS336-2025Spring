import sys
import os
import random

from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.language_identification import identify_language

# 配置路径
INPUT_WET = "cs336_data/data/sample_negative.warc.wet.gz"
OUTPUT_NEG_FILE = "cs336_data/data/negative_train_data.txt"

# 设置目标数量（建议与正样本数量保持 1:1 或 1:2）
TARGET_COUNT = 20000

def process_negative_samples(input_path, output_path, target_count):
    count = 0
    kept = 0
    
    print(f"开始处理负样本，目标数量: {target_count}...")

    with open(output_path, "w", encoding="uft-8") as out_f:
        for record in ArchiveIterator(open(input_path, "rb"), record_types=WarcRecordType.conversion):
            if kept >= target_count:
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
                out_f.write(f"__label__low_quality {clean_text}\n")
                kept += 1
            
            except Exception as e:
                print(f"Error processing record: {e}")
                continue
            
            count += 1

            if count % 1000 == 0:
                print(f"Processed {count} records, kept {kept} high-quality examples.")

        print(f"Finished! Total processed: {count}, Total kept: {kept}")

if __name__ == "__main__":
    process_negative_samples(INPUT_WET, OUTPUT_NEG_FILE, TARGET_COUNT)