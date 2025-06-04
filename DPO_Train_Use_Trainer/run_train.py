import torch
import argparse
import pandas as pd
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./dpo_model_output', help='模型保存路径')
    parser.add_argument('--train_data_path', type=str, default='./data/train-00000-of-00001-789dc5dece0f1fc1.parquet', help='训练集')
    parser.add_argument('--eval_data_path', type=str, default='./data/test-00000-of-00001-8ecd46436fadcf7f.parquet', help='测试集')
    parser.add_argument('--pretrain_model', type=str, default='../Qwen2.5-0.5B-Instruct', help='预训练模型')
    parser.add_argument('--num_epochs', type=int, default=3, help='训练多少轮')
    parser.add_argument('--batch_size', type=int, default=2, help='训练批次大小')
    parser.add_argument('--max_len', type=int, default=300, help='最大长度')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--logging_steps', type=int, default=5, help='日志打印步数')
    args = parser.parse_args()
    return args


def load_data(path, is_train=True):
    df = pd.read_parquet(path)
    all_data = []
    for prompt, chosen, rejected in zip(df['prompt'], df['chosen'], df['rejected']):
        temp = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
        all_data.append(temp)
    if is_train:
        all_data = all_data[:20000]  # 只取前1000条数据
    else:
        all_data = all_data[:2000] 
    return all_data


def preprocess_function(examples):
    inputs = tokenizer(examples["prompt"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    chosen_outputs = tokenizer(examples["chosen"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    rejected_outputs = tokenizer(examples["rejected"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "chosen_ids": chosen_outputs["input_ids"],
        "rejected_ids": rejected_outputs["input_ids"]
    }


def create_model():
    model = AutoModelForCausalLM.from_pretrained(args.pretrain_model)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model)
    return model, tokenizer


if __name__ == '__main__':
    args = set_args()

    train_data = load_data(args.train_data_path)
    eval_data = load_data(args.eval_data_path)
    print("训练集:{}, 验证集:{}".format(len(train_data), len(eval_data)))  # 训练集:19862, 验证集:4996

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = create_model()
    model_ref, _ = create_model()
    
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True)

    training_args = DPOConfig(
        output_dir="DPO_model_output",
        # eval_strategy="epoch",     # 每个 epoch 进行一次评估
        eval_strategy="steps",
        eval_steps=1000,
        
        learning_rate=2e-5,         # 学习率
        per_device_train_batch_size=2,       # 每个设备的训练批量大小
        per_device_eval_batch_size=2,        # 每个设备的验证批量大小
        num_train_epochs=2,   # 训练的 epoch 数
        weight_decay=0.01,    # 权重衰减
        logging_dir='DPO_model_output',
        logging_steps=10,                    # 每 10 步记录一次日志
        # save_strategy="epoch",               # 每个 epoch 保存一次模型
        save_strategy="steps",        # 改为每隔一定步数保存模型
        save_steps=1000, 
        load_best_model_at_end=True,         # 在训练结束时加载最佳模型
        save_total_limit=2,                  # 最多保存两个检查点
        fp16=True,                           # 使用混合精度训练（需要 GPU）
        dataloader_num_workers=2,           # 数据加载的线程数
        max_prompt_length=100,    # prompt统一padding多长
        max_completion_length=200   # 回答padding多长
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
    trainer.save_model(args.output_dir)
    print("模型训练完成，保存路径：", args.output_dir)

