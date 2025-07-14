"""
@file   : data_helper.py
@time   : 2025-04-27
"""
import json
import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        super().__init__()
        self.file_path = file_path
        self.examles = self._load_data(self.file_path)
        self.tokenizer = tokenizer

    def _load_data(self, file_path):
        items = []
        with open(file_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = json.loads(line)
                items.append(line)
        return items

    def __len__(self):
        return len(self.examles)

    def __getitem__(self, index):
        example = self.examles[index]
        dialog = [{"role": "system", "content": "You are a helpful asssistant."},
                  {"role": "user", "content": example['query']},
                  {"role": "assistant", "content": example['answer']}]
        chat = self.tokenizer.apply_chat_template(dialog, tokenize=False)
        '''
        <|im_start|>system\nYou are a helpful asssistant.<|im_end|>\n
        <|im_start|>user\n保持健康的三个提示。<|im_end|>\n
        <|im_start|>assistant\n以下是保持健康的三个提、高脂肪和加工食品，以保持健康的饮记忆力。<|im_end|>\n'
        '''
        return chat


def sft_collate(batch, tokenizer, end_str, max_length):
    inputs = tokenizer(batch, max_length=max_length, padding=True, truncation=True)
    input_ids = inputs['input_ids']
    input_len = len(input_ids[0])

    end_ids = tokenizer(end_str)['input_ids']
    end_id_len = len(end_ids)
    loss_mask = []
    for input_id in input_ids:
        for i in range(len(input_id) - end_id_len, -1, -1):
            # 生成掩码，忽略 |im_start|>system 和 |im_end| 之前的部分
            # 说白了 只计算回答部分的损失。 问题不计算损失
            if input_id[i:i + end_id_len] == end_ids:
                mask = [1] * input_len
                mask[:i + end_id_len] = [0] * (i + end_id_len)
                loss_mask.append(mask)
                break
            if i == 0:
                # 如果没有找到 |im_end|，则整个序列都不计算损失
                loss_mask.append([0] * input_len)
    # print(inputs)  #  {"input_ids": ..., "attention_mask": ...}
    # print(loss_mask)

    inputs = {k: torch.tensor(v) for k, v in inputs.items()}
    loss_mask = torch.tensor(loss_mask)
    return inputs, loss_mask