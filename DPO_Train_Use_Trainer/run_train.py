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



'''
{'loss': 1.0619, 'grad_norm': 130.79339599609375, 'learning_rate': 1.9971134159030648e-05, 'rewards/chosen': -1.5285903215408325, 'rewards/rejected': -1.4812065362930298, 'rewards/accuracies': 0.4000000059604645, 'rewards/margins': -0.047383688390254974, 'logps/chosen': -199.06910705566406, 'logps/rejected': -173.42794799804688, 'logits/chosen': -0.24395839869976044, 'logits/rejected': -0.1604161560535431, 'epoch': 0.01}

在使用 DPOTrainer 训练时，每步打印的信息是训练过程中记录的各种指标和状态。这些信息可以帮助你监控训练过程，分析模型的性能和优化状态。以下是每个字段的含义解释：
字段解释
loss:
含义：当前训练步的损失值。
作用：损失值是优化的目标，通常会随着训练逐步下降。如果损失值异常（如 NaN 或过大），可能需要检查数据或模型的数值稳定性。

grad_norm:
含义：当前训练步的梯度范数（通常是 L2 范数）。
作用：梯度范数反映了梯度的大小。如果梯度过大，可能导致梯度爆炸；如果梯度过小，可能导致训练停滞。可以通过梯度裁剪（如 torch.nn.utils.clip_grad_norm_）来控制梯度大小。

learning_rate:
含义：当前训练步的学习率。
作用：学习率是优化器的重要参数，控制模型权重更新的步长。如果使用学习率调度器（如 lr_scheduler），学习率可能会动态变化。

rewards/chosen:
含义：模型对 chosen（被选中的样本）的奖励值。
作用：奖励值通常由奖励函数计算，用于衡量模型对选中样本的偏好。

rewards/rejected:
含义：模型对 rejected（被拒绝的样本）的奖励值。
作用：奖励值用于衡量模型对被拒绝样本的偏好。理想情况下，rewards/chosen 应该高于 rewards/rejected。

rewards/accuracies:
含义：当前训练步的准确率。
作用：准确率表示模型在当前训练步中对样本的正确选择比例。通常用于评估模型的性能。

rewards/margins:
含义：rewards/chosen 和 rewards/rejected 的差值（即奖励的边距）。
作用：奖励边距反映了模型对 chosen 和 rejected 样本的区分能力。理想情况下，边距应该为正且逐步增大。

logps/chosen:
含义：chosen 样本的对数概率（log-probability）。
作用：对数概率是模型对 chosen 样本的预测置信度的对数形式。值越高，表示模型对该样本的预测越自信。

logps/rejected:
含义：rejected 样本的对数概率。
作用：与 logps/chosen 类似，但针对被拒绝样本。理想情况下，logps/chosen 应该高于 logps/rejected。

logits/chosen:
含义：chosen 样本的原始模型输出（logits）。
作用：logits 是模型在最后一层的原始输出，未经过 softmax 转换。值越高，表示模型对该样本的偏好越强。

logits/rejected:
含义：rejected 样本的原始模型输出（logits）。
作用：与 logits/chosen 类似，但针对被拒绝样本。

epoch:
含义：当前训练的 epoch（轮次）。
作用：表示训练的进度。epoch 通常是一个小数，表示当前训练步在整个 epoch 中的比例。
'''
