import torch
import argparse
import pandas as pd
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments


# torch=2.3.0+cu121
# transformers=4.5.1
# trl=0.17.0
# datasets=3.6.0
def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./dpo_model_output', help='模型保存路径')
    parser.add_argument('--train_data_path', type=str, default='./data/train-00000-of-00001-789dc5dece0f1fc1.parquet', help='训练集')
    parser.add_argument('--eval_data_path', type=str, default='./data/test-00000-of-00001-8ecd46436fadcf7f.parquet', help='测试集')
    parser.add_argument('--pretrain_model', type=str, default='./Qwen2.5-0.5B-Instruct', help='预训练模型')
    parser.add_argument('--num_epochs', type=int, default=5, help='训练多少轮')
    parser.add_argument('--batch_size', type=int, default=2, help='训练批次大小')
    parser.add_argument('--max_len', type=int, default=300, help='最大长度')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--logging_steps', type=int, default=5, help='日志打印步数')
    args = parser.parse_args()
    return args


def load_data(path):
    df = pd.read_parquet(path)
    all_data = []
    for prompt, chosen, rejected in zip(df['prompt'], df['chosen'], df['rejected']):
        temp = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
        all_data.append(temp)
    return all_data


def preprocess_function(examples):
    # 因为DPOTrainer会给每条文本后面加eos_token_id  所以把<|im_end|>可以去掉
    prompt = [{"role": "user", "content": examples['prompt']}]
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
    # print(prompt)
    '''
    <|im_start|>system
    You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
    <|im_start|>user
    今年我想在室内种植水果，你能给我推荐一个容易入手的水果吗？<|im_end|>
    '''
    prompt = prompt.rstrip('<|im_end|>\n')
    # print(tokenizer.eos_token)  # <|im_end|>
    
    # 将chosen和rejected 只需要处理成'<|im_start|>assistant\n文本'会将两个文本直接映射为id 相连
    chosen = "<|im_start|>assistant\n{}".format(examples['chosen'])
    rejected = "<|im_start|>assistant\n{}".format(examples['rejected'])
    return {
        "prompt": prompt,
        'chosen': chosen,
        'rejected': rejected
    }


def create_model():
    model = AutoModelForCausalLM.from_pretrained(args.pretrain_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model)
    return model, tokenizer


if __name__ == '__main__':
    args = set_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = create_model()
    model_ref, _ = create_model()

    train_data = load_data(args.train_data_path)
    eval_data = load_data(args.eval_data_path)

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    train_dataset = train_dataset.map(preprocess_function, batched=False)  # 这里的batched=False 是因为实现的preprocess_function函数是单样本处理
    # print(train_dataset[100])
    eval_dataset = eval_dataset.map(preprocess_function, batched=False)

    training_args = DPOConfig(
        output_dir="DPO_model_output",
        eval_strategy="epoch",     # 每个 epoch 进行一次评估
        learning_rate=2e-5,         # 学习率
        per_device_train_batch_size=2,       # 每个设备的训练批量大小
        per_device_eval_batch_size=2,        # 每个设备的验证批量大小
        num_train_epochs=3,   # 训练的 epoch 数
        weight_decay=0.01,    # 权重衰减
        logging_dir='DPO_model_output',
        logging_steps=10,                    # 每 10 步记录一次日志
        save_strategy="epoch",               # 每个 epoch 保存一次模型
        load_best_model_at_end=True,         # 在训练结束时加载最佳模型
        save_total_limit=2,                  # 最多保存两个检查点
        fp16=True,                           # 使用混合精度训练（需要 GPU）
        dataloader_num_workers=2,           # 数据加载的线程数
        max_prompt_length=200,    # prompt统一padding多长
        max_completion_length=500   # 回答padding多长
    )

    # 初始化 DPOTrainer
    trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer
    )
    trainer.train()
