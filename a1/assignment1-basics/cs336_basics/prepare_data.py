import os
import numpy as np
import pickle
from .tokenizer import train_bpe, Tokenizer

# ---------------- 配置部分 ----------------
# 原始数据路径
TRAIN_TEXT_PATH = "data/TinyStoriesV2-GPT4-train.txt"
VAL_TEXT_PATH = "data/TinyStoriesV2-GPT4-valid.txt"

# 输出文件路径
TOKENIZER_MODEL_PATH = "data/tokenizer_model.pkl" # 保存训练好的分词器
TRAIN_BIN_PATH = "data/tinystories_train.bin"     # 训练集二进制文件
VAL_BIN_PATH = "data/tinystories_val.bin"         # 验证集二进制文件

# 参数配置 
VOCAB_SIZE = 10000
SPECIAL_TOKENS = ["<|endoftext|>"]
# ----------------------------------------

def prepare():
    # --- 创建训练子集 ---
    # 定义子集的大小， 10MB (10 * 1024 * 1024 字节)
    SUBSET_SIZE = 10 * 1024 * 1024 
    SUBSET_PATH = "data/train_subset_for_bpe.txt"
    
    print(f"---  准备 BPE 训练子集 (使用前 {SUBSET_SIZE / (1024*1024):.1f} MB 数据) ---")
    
    # 从原始训练集读取开头的一部分数据
    if os.path.exists(TRAIN_TEXT_PATH):
        with open(TRAIN_TEXT_PATH, "r", encoding="utf-8") as f:
            subset_data = f.read(SUBSET_SIZE)
        
        # 写入临时文件
        with open(SUBSET_PATH, "w", encoding="utf-8") as f:
            f.write(subset_data)
        print(f"已创建临时训练文件: {SUBSET_PATH}")
    else:
        print(f"错误: 找不到文件 {TRAIN_TEXT_PATH}")
        return

    # ----------------------------------------
    print(f"--- 开始训练 BPE 分词器 (Vocab Size: {VOCAB_SIZE}) ---")
    vocab, merges = train_bpe(SUBSET_PATH, VOCAB_SIZE, SPECIAL_TOKENS)
    print(f"分词器训练完成! Vocab size: {len(vocab)}")
    print(f"Merges count: {len(merges)}")
    
    print(f"保存分词器到: {TOKENIZER_MODEL_PATH}")
    with open(TOKENIZER_MODEL_PATH, 'wb') as f:
        pickle.dump({
            "vocab": vocab,
            "merges": merges,
            "special_tokens": SPECIAL_TOKENS
        }, f)

    # 实例化分词器
    enc = Tokenizer(vocab, merges, SPECIAL_TOKENS)
    # ----------------------------------------
    print("\n--- 编码并保存训练集 ---")
    encode_and_save(enc, TRAIN_TEXT_PATH, TRAIN_BIN_PATH)

    # ----------------------------------------
    print("\n--- 编码并保存验证集 ---")
    encode_and_save(enc, VAL_TEXT_PATH, VAL_BIN_PATH)
    
    print("\n所有预处理完成!")

def encode_and_save(tokenizer, text_path, bin_path):
    """读取文本,编码为ID,保存为uint16 numpy array"""
    if not os.path.exists(text_path):
        print(f"警告: 文件 {text_path} 不存在，跳过。")
        return

    print(f"正在读取 {text_path} ...")
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"正在编码 {len(text)} 字符...")
    ids = tokenizer.encode(text)
    print(f"编码完成，共 {len(ids)} 个 token。")

    print(f"正在保存到 {bin_path} (dtype=uint16)...")
    # 作业要求使用 uint16 来节省空间 (因为 vocab_size=10000 < 65535)
    ids = np.array(ids, dtype=np.uint16)
    with open(bin_path, 'wb') as f:
        f.write(ids.tobytes())
    print("保存成功。")

if __name__ == "__main__":
    prepare()