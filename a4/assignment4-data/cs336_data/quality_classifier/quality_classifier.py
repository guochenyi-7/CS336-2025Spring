import fasttext
import os
import re

# 获取当前脚本文件的绝对路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 构建模型文件的相对路径
# 根据截图：从 quality_classifier/ 向上两级到 cs336_data/，然后进入 models/
MODEL_PATH = os.path.join(CURRENT_DIR, "../models/quality_classifier.bin")

try:
    model = fasttext.load_model(MODEL_PATH)
    print(f"成功加载模型: {MODEL_PATH}")
except Exception as e:
    print(f"无法加载模型，请检查路径: {MODEL_PATH}")
    raise e

def classify_quality(text: str):
    if not text:
        return ("unknow", 0.0)
    
    clean_text = text.replace("\n", " ").strip()
    labels, scores = model.predict(clean_text)
    label = labels[0]
    score = scores[0]

    if label.startswith("__label__"):
        label = label.replace("__label__", "")
    
    return label, score
