import fasttext
import os

from warcio.archiveiterator import ArchiveIterator
from cs336_data.extract_text_from_html_bytes import extract_text_from_html_bytes
# 获取当前脚本所在的文件夹路径
curr_dir = os.path.dirname(os.path.abspath(__file__))

# 拼接出模型的绝对路径
MODEL_PATH = os.path.join(curr_dir, "models", "lid.176.bin")

# 全局加载模型，避免函数重复加载开销
model = None
if os.path.exists(MODEL_PATH):
    model = fasttext.load_model(MODEL_PATH)
else:
    print(f"Warning: Model file not found at {MODEL_PATH}. Please download it.")

def identify_language(text: str) -> tuple[str, float]:
    """
    识别给定字符串的主要语言。
    
    Args:
        text: Unicode 字符串
        
    Returns:
        tuple: (语言标识符, 置信度 0-1)
    """
    if model is None:
        raise RuntimeError(f"FastText model not loaded. Check path: {MODEL_PATH}")
    
    # 移除换行符，因为 fastText 这里是按行处理的
    clean_text = text.replace('\n', ' ').strip()
    
    if not clean_text:
        return ("unknown", 0.0)

    # predict 返回格式通常为 (['__label__en'], [0.98])
    labels, scores = model.predict(clean_text, k=1)
    
    label = labels[0]
    score = float(scores[0])
    
    # 移除 fastText 的 label 前缀
    lang_id = label.replace('__label__', '')
    
    return lang_id, score

def analyze_language_identification(warc_path, num_samples=20):
    total_docs = 0
    english_docs = 0
    samples = []
    
    print(f"Processing {warc_path}...")
    
    with open(warc_path, 'rb') as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type != 'response':
                continue
            # 获取 HTML 内容
            content_bytes = record.content_stream().read()
            # 提取文本
            text = extract_text_from_html_bytes(content_bytes)

            if not text or len(text.strip()) < 50: # 跳过过短的文本
                continue
                
            # 语言识别
            lang, score = identify_language(text)
            
            total_docs += 1
            if lang == 'en':
                english_docs += 1
            
            # 收集样本用于手动检查
            if len(samples) < num_samples:
                samples.append({
                    'text_snippet': text[:200].replace('\n', ' '), # 只看前200字符
                    'predicted_lang': lang,
                    'score': score
                })
    
    # 打印结果
    print(f"\nTotal Documents Scanned: {total_docs}")
    print(f"English Fraction: {english_docs / total_docs:.2%}")
    
    print(f"\n--- Manual Inspection of {num_samples} Samples ---")
    for i, sample in enumerate(samples):
        print(f"[{i+1}] Pred: {sample['predicted_lang']} (Score: {sample['score']:.4f}) | Text: {sample['text_snippet']}")

if __name__ == "__main__":
    # 获取当前脚本所在的文件夹路径
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 使用文件名
    warc_filename = 'CC-MAIN-20250417135010-20250417165010-00065.warc.gz'

    # 拼接完整路径
    warc_path = os.path.join(current_dir, warc_filename)
    
    analyze_language_identification(warc_path, num_samples=20)