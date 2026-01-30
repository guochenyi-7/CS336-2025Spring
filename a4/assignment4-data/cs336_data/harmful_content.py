import os
import fasttext

# 获取当前脚本所在的文件夹路径
curr_dir = os.path.dirname(os.path.abspath(__file__))

# 拼接出模型的绝对路径
MODEL_NSFW_PATH = os.path.join(curr_dir, "models", "jigsaw_fasttext_bigrams_nsfw_final.bin")

# 全局加载模型，避免函数重复加载开销
model_nsfw = None
if os.path.exists(MODEL_NSFW_PATH):
    model_nsfw = fasttext.load_model(MODEL_NSFW_PATH)
else:
    print(f"Warning: Model file not found at {MODEL_NSFW_PATH}. Please download it.")

def classify_nsfw(text: str):
    if model_nsfw is None:
        raise RuntimeError(f"FastText model not loaded. Check path: {MODEL_NSFW_PATH}")
    
    # 移除换行符，因为 fastText 这里是按行处理的
    clean_text = text.replace('\n', ' ').strip()
    
    if not clean_text:
        return ("unknown", 0.0)

    labels, scores = model_nsfw.predict(clean_text)

    label = labels[0]
    score = scores[0]
    
    if label == "__label__nsfw":
        label = "nsfw"
    else:
        label = "non-nsfw"

    return label, score


# 获取当前脚本所在的文件夹路径
curr_dir = os.path.dirname(os.path.abspath(__file__))

# 拼接出模型的绝对路径
MODEL_HS_PATH = os.path.join(curr_dir, "models", "jigsaw_fasttext_bigrams_hatespeech_final.bin")

# 全局加载模型，避免函数重复加载开销
model_hs = None
if os.path.exists(MODEL_HS_PATH):
    model_hs = fasttext.load_model(MODEL_HS_PATH)
else:
    print(f"Warning: Model file not found at {MODEL_HS_PATH}. Please download it.")

def classify_toxic_speech(text: str):
    if model_hs is None:
        raise RuntimeError(f"FastText model not loaded. Check path: {MODEL_HS_PATH}")
    
    # 这里转换成大写即可成功
    clean_text = text.replace('\n', ' ').strip().upper()
    
    if not clean_text:
        return ("unknown", 0.0)

    labels, scores = model_hs.predict(clean_text)

    label = labels[0]
    score = scores[0]
    # print(f"\n[DEBUG] Raw label: {label}, Score: {scores[0]}")
    threshold = 0.7
    final_label = "non-toxic" 

    if label == "__label__toxic":
        if score > threshold:
            final_label = "toxic"
        else:
            final_label = "non-toxic"
    else:
        final_label = "non-toxic"
    
    return final_label, score
