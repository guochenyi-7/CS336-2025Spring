import gzip
import os

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
from fastwarc.warc import ArchiveIterator, WarcRecordType

def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    """
    从HTML字节流中提取纯文本
    """
    encoding = "utf-8"
    decoded_html = ""

    try:
        decoded_html = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        detected_encoding = detect_encoding(html_bytes)

        if detected_encoding:
            encoding = detected_encoding
        
        try:
            decoded_html = html_bytes.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            decoded_html = html_bytes.decode("utf-8", error="replace")

    return extract_plain_text(decoded_html, main_content=False, alt_texts=True)

def main():
   # 获取当前脚本所在的文件夹路径
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 使用文件名
    warc_filename = 'CC-MAIN-20250417135010-20250417165010-00065.warc.gz'
    wet_filename = 'CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz'

    # 拼接完整路径
    warc_path = os.path.join(current_dir, warc_filename)
    wet_path = os.path.join(current_dir, wet_filename)

    # 读取 WARC 文件并提取文本
    print(f"正在读取 WARC 文件: {warc_path}")
    warc_texts = {}
    with gzip.open(warc_path, 'rb') as stream:
        # 遍历 WARC 文件中的记录
        for record in ArchiveIterator(stream, record_types=WarcRecordType.response):
            url = record.headers.get('WARC-Target-URI')
            # 获取原始 HTML 二进制数据
            html_bytes = record.reader.read()
            
            my_text = extract_text_from_html_bytes(html_bytes)
            warc_texts[url] = my_text
            
            # 为了演示，这里只取前几个例子，避免内存爆炸
            if len(warc_texts) >= 10: 
                break

    # 读取 WET 文件并获取官方文本
    print(f"正在读取 WET 文件: {wet_path}")
    wet_texts = {}
    with gzip.open(wet_path, 'rb') as stream:
        for record in ArchiveIterator(stream, record_types=WarcRecordType.conversion):
            url = record.headers.get('WARC-Target-URI')
            if url in warc_texts:
                # WET 文件的内容已经是提取好的文本，可以直接读取
                # 注意：WET 内容通常是 bytes，需要 decode
                wet_content = record.reader.read().decode('utf-8', errors='replace')
                wet_texts[url] = wet_content

    # 对比打印
    for url in warc_texts:
        if url in wet_texts:
            print(f"=== URL: {url} ===")
            print("--- [MY EXTRACTION] ---")
            print(warc_texts[url][:500]) # 打印前500个字符预览
            print("\n--- [WET EXTRACTION] ---")
            print(wet_texts[url][:500])
            print("\n" + "="*30 + "\n")

if __name__ == "__main__":
    main()