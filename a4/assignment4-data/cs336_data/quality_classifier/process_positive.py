import os
import sys
from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract_text_from_html_bytes import extract_text_from_html_bytes
from cs336_data.language_identification import identify_language
from cs336_data.gopher_quality_filters import gopher_quality_filter

INPUT_WARC = "cs336_data/data/positive_samples.warc.gz"
OUTPUT_TRAIN_FILE = "cs336_data/data/positive_train_data.txt"

def process_warc(input_path, output_path):
    count = 0
    kept = 0

    with open(output_path, "w", encoding="uft-8") as out_f:
        for record in ArchiveIterator(open(input_path, "rb"), record_types=WarcRecordType.response):
            try:
                html_bytes = record.reader.read()
                text = extract_text_from_html_bytes(html_bytes)
                if not text:
                    continue

                # 语言过滤
                lang, l_score = identify_language(text)
                if lang != "en" or l_score < 0.6:
                    continue

                # Gopher 质量过滤
                if gopher_quality_filter(text) == False:
                    continue

                clean_text = text.replace("\n", ' ').replace("\r", ' ')
                out_f.write(f"__label__high_quality {clean_text}\n")

                kept += 1

            except Exception as e:
                print(f"Error processing record: {e}")
                continue
            
            count += 1

            if count % 1000 == 0:
                print(f"Processed {count} records, kept {kept} high-quality examples.")

        print(f"Finished! Total processed: {count}, Total kept: {kept}")

if __name__ == "__main__":
    process_warc(INPUT_WARC, OUTPUT_TRAIN_FILE)
    