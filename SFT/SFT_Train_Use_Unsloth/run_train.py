import json
import torch
import os
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer


# unsloth=2025.4.7
# torch='2.3.0+cu121'
# transformers='4.51.3'
# trl='0.17.0'
# datasets='3.6.0'

class QwenDataset(Dataset):
    def __init__(self, file_path, tokenizer, end_str, max_length=300):
        """
        自定义数据集，用于加载 Qwen 微调任务的数据。
        :param file_path: 数据文件路径
        :param tokenizer: 分词器
        :param max_length: 最大序列长度
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.end_str = end_str

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                self.samples.append(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        dialog = [{"role": "system", "content": "You are a helpful asssistant."},
                  {"role": "user", "content": sample['content']},
                  {"role": "assistant", "content": sample['summary']}]
        chat = self.tokenizer.apply_chat_template(dialog, tokenize=False)

        inputs = self.tokenizer(chat)   # 在这里不填充
        
        input_ids = inputs['input_ids']
        input_len = len(input_ids)

        end_ids = tokenizer(self.end_str)['input_ids']
        end_id_len = len(end_ids)

        for i in range(len(input_ids) - end_id_len, -1, -1):
            if input_ids[i:i+end_id_len] == end_ids:
                labels = input_ids.copy()
                labels[:i+end_id_len] = [-100] * (i+end_id_len)
                break
            if i == 0:
                labels = [-100] * input_len

        # 消除padding影响
        for i in range(len(labels)):
            if labels[i] == self.tokenizer.pad_token_id:
                labels[i] = -100        

        inputs['labels'] = labels    # 包含三个部分: input_ids  attention_mask以及labels
        return {key: torch.tensor(val, dtype=torch.long) for key, val in inputs.items()}


if __name__ == '__main__':
    # 使用unsloth 唯一不同 就是这里
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "./Qwen2.5-0.5B-Instruct",
        # max_seq_length = 300,   # Context length - can be longer, but uses more memory
        load_in_4bit = True,     # 4bit uses much less memory
        load_in_8bit = False,    # A bit more accurate, uses 2x memory
        full_finetuning = False, # We have full finetuning now!
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,  # Best to choose alpha = rank or rank*2
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,   # We support rank stabilized LoRA
        loftq_config = None,  # And LoftQ
    )

    train_dataset = QwenDataset("./data/train.json", tokenizer, end_str='<|im_start|>assistant\n')
    eval_dataset = QwenDataset("./data/test.json", tokenizer, end_str='<|im_start|>assistant\n')

    output_dir = './sft_model_output'
        
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,          # 保存模型和日志的目录
        eval_strategy='epoch',    # 每个 epoch 进行一次评估
        learning_rate=5e-5,                  # 学习率
        per_device_train_batch_size=2,       # 每个设备的训练批量大小
        per_device_eval_batch_size=2,        # 每个设备的验证批量大小
        num_train_epochs=3,                  # 训练的 epoch 数
        weight_decay=0.01,                   # 权重衰减
        logging_dir=output_dir,           # 日志目录
        logging_steps=10,                    # 每 10 步记录一次日志
        save_strategy="epoch",               # 每个 epoch 保存一次模型
        load_best_model_at_end=True,         # 在训练结束时加载最佳模型
        save_total_limit=2,                  # 最多保存两个检查点
        # fp16=True,                           # 使用混合精度训练（需要 GPU）
        dataloader_num_workers=4,            # 数据加载的线程数
    )

    # 定义 Trainer
    trainer = Trainer(
        model=model,                         # 要训练的模型
        args=training_args,                  # 训练参数
        train_dataset=train_dataset,         # 训练数据集
        eval_dataset=eval_dataset,           # 验证数据集
        tokenizer=tokenizer,                 # 分词器
        data_collator=DataCollatorForSeq2Seq(tokenizer)
    )

    # 开始训练
    trainer.train()

    # 评估模型
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # 保存模型
    trainer.save_model(os.path.join(output_dir, 'sft_best_model'))
    

    

