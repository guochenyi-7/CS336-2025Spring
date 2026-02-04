import os
import hashlib

from collections import Counter

def exact_line_deduplication(input_files, output_dir):
    line_counters = Counter()

    # 第一遍扫描统计哈希值
    for filepath in input_files:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                clean_line = line.strip()
                if not clean_line:
                    continue

                line_hash = hashlib.sha256(clean_line.encode("utf-8")).digest()
                line_counters[line_hash] += 1


    # 第二遍去重
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for filepath in input_files:
        filename = os.path.basename(filepath)
        final_output_path = os.path.join(output_dir, filename)
        with open(filepath, "r", encoding="utf-8", errors='ignore') as fin, \
             open(final_output_path, "w", encoding="utf-8") as fout:
            
            for line in fin:
                clean_line = line.strip()
                if not clean_line:
                    continue

                line_hash = hashlib.sha256(clean_line.encode("utf-8")).digest()
                if line_counters[line_hash] == 1:
                    fout.write(line)
    