import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./dpo_model_output', help='模型保存路径')
    parser.add_argument('--train_data_path', type=str, default='./data/train-00000-of-00001-789dc5dece0f1fc1.parquet', help='训练集')
    parser.add_argument('--test_data_path', type=str, default='./data/test-00000-of-00001-8ecd46436fadcf7f.parquet', help='测试集')
    parser.add_argument('--pretrain_model', type=str, default='../../Qwen2.5-1.5B-Instruct', help='预训练模型')
    parser.add_argument('--num_epochs', type=int, default=5, help='训练多少轮')
    parser.add_argument('--batch_size', type=int, default=2, help='训练批次大小')
    parser.add_argument('--max_len', type=int, default=300, help='最大长度')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--logging_steps', type=int, default=5, help='日志打印步数')
    args = parser.parse_args()
    return args
