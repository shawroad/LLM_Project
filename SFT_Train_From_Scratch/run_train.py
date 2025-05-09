"""
@file   : run_train.py
@time   : 2025-04-27
"""
import os
import torch
import json
import math
import random
import argparse
import functools
import numpy as np
from config import set_args
from torch.utils.data import DataLoader
from data_helper import SFTDataset, sft_collate
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboard import SummaryWriter

# Qwen2-7B-Instruct: 28层


def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def evaluate_ppl(model, dataloader, loss_function):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for inputs, loss_mask in dataloader:
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                loss_mask = loss_mask.cuda()

            outputs = model(**inputs)
            logits = outputs.logits[:, :-1, :]
            input_ids = inputs['input_ids']
            labels = input_ids[:, 1:]
            loss_mask = loss_mask[:, 1:]

            logits = logits.reshape(-1, logits.size(-1))
            labels = labels.reshape(-1)
            loss_mask = loss_mask.reshape(-1)

            loss = loss_function(logits, labels)
            total_loss += torch.sum(loss * loss_mask).item()
            total_tokens += torch.sum(loss_mask).item()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl


if __name__ == '__main__':
    args = set_args()
    set_seed()

    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model, use_fast=False)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token

    # 回调函数
    collate_fn = functools.partial(sft_collate,
                                   tokenizer=tokenizer,
                                   end_str="<|im_start|>assistant\n",
                                   max_length=args.max_len)

    train_dataset = SFTDataset(args.train_data_path, tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    valid_dataset = SFTDataset(args.test_data_path, tokenizer=tokenizer)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 模型加载为 4 位量化格式
        bnb_4bit_use_double_quant=True,  # 双重量化是一种优化技术，先对权重进行一次粗量化，然后在计算时再进行更精细的量化。
        bnb_4bit_quant_type='nf4',  # nf4 是一种 4 位量化格式，通常用于减少模型大小和加速推理。
        bnb_4bit_compute_dtype=torch.bfloat16  # 计算时使用 bfloat16 精度
    )

    # model = AutoModelForCausalLM.from_pretrained(args.pretrain_model, quantization_config=bnb_config)
    model = AutoModelForCausalLM.from_pretrained(args.pretrain_model, torch_dtype=torch.bfloat16, device_map='auto')
    # print(model)
    print(f'模型总参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    # 模型总参数量为：1543714304

    peft_config = LoraConfig(
        r=8,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj'],
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=16,
        lora_dropout=0.05
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    # trainable params: 5,046,272 || all params: 7,620,662,784 || trainable%: 0.06621828235983522
    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    best_ppl = float('inf')  # ppl越小越好
    tb_write = SummaryWriter(log_dir=args.output_dir)
    global_step, tr_loss, logging_loss = 0, 0, 0
    for epoch in range(args.num_epochs):
        model.train()
        for step, (inputs, loss_mask) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                loss_mask = loss_mask.cuda()
            # print(inputs['input_ids'].size())   # torch.Size([4, 50])
            # print(inputs['attention_mask'].size())  # torch.Size([4, 50])
            outputs = model(**inputs)
            # print(outputs.logits.size())   # torch.Size([4, 50, 152064])

            logits = outputs.logits[:, :-1, :]
            # print(logits.size())      # torch.Size([4, 49, 152064])

            input_ids = inputs['input_ids']
            # print(input_ids.size())   # torch.Size([4, 50])

            labels = input_ids[:, 1:]
            loss_mask = loss_mask[:, 1:]
            # print(loss_mask.size())   # torch.Size([4, 49])

            logits = logits.reshape(-1, logits.size(-1))
            labels = labels.reshape(-1)
            loss_mask = loss_mask.reshape(-1)

            loss = loss_function(logits, labels)

            loss = torch.sum(loss * loss_mask) / loss_mask.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"epoch:{epoch}, step: {step}, loss: {loss.item()}")

            global_step += 1
            tr_loss += loss.item()
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                tb_write.add_scalar("train_loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss

            if global_step % 5000 == 0:
                # 验证集评估
                valid_ppl = evaluate_ppl(model, valid_dataloader, loss_function)
                print(f"epoch: {epoch}, valid ppl: {valid_ppl}")
                if valid_ppl < best_ppl:
                    best_ppl = valid_ppl
                    model.save_pretrained(os.path.join(args.output_dir, 'sft_lora_model_best'))

                model.save_pretrained(
                    os.path.join(args.output_dir, 'sft_lora_model_{}_step_{}'.format(epoch, global_step)))
                with open(os.path.join(args.output_dir, 'logs.txt'), 'a', encoding='utf8') as f:
                    f.write(f"epoch: {epoch}, valid ppl: {valid_ppl}\n")

