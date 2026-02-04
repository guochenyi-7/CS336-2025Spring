import fasttext
import os

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义文件路径
# 注意：这里假设 train.txt 就在脚本同级的 data 目录下
train_data_path = os.path.join(current_dir, 'data', 'train.txt')
model_output_path = os.path.join(current_dir, '..', 'models', 'quality_classifier.bin')

os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

def train():
    if not os.path.exists(train_data_path):
        print(f"Error: 找不到训练数据 {train_data_path}")
        return

    print(f"开始训练模型，使用数据: {train_data_path} ...")

    model = fasttext.train_supervised(
        input=train_data_path,
        lr=0.5,
        epoch=5,
        wordNgrams=2,
        bucket=200000,
        dim=50,
        loss='softmax'
    )  

    # 保存模型
    model.save_model(model_output_path)
    print(f"训练完成！模型已保存至: {model_output_path}")

if __name__ == "__main__":
    train()

