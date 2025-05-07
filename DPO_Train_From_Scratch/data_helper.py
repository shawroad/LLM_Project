import torch
from torch.utils.data import Dataset


class DPODataset(Dataset):
    def __init__(self, data, tokenizer):
        super().__init__()
        self.data = data
        self.prompt = self.data['prompt'].tolist()
        self.chosen = self.data['chosen'].tolist() 
        self.rejected = self.data['rejected'].tolist() 
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        prompt = self.prompt[index]
        chosen = self.chosen[index]
        rejected = self.rejected[index]
        chosen_dialog = [{"role": "system", "content": "You are a helpful asssistant."},
                         {"role": "user", "content": prompt},
                         {"role": "assistant", "content": chosen}]

        chosen_dialog = self.tokenizer.apply_chat_template(chosen_dialog, tokenize=False)

        rejected_dialog = [{"role": "system", "content": "You are a helpful asssistant."},
                           {"role": "user", "content": prompt},
                           {"role": "assistant", "content": rejected}]
        rejected_dialog = self.tokenizer.apply_chat_template(rejected_dialog, tokenize=False)

        return {"chosen_dialog": chosen_dialog, 'rejected_dialog': rejected_dialog}


def get_target_ids(input_ids, end_id_len, end_ids, tokenizer):
    input_len = len(input_ids[0])
    target_ids = []
    for input_id in input_ids:
        for i in range(len(input_id)-end_id_len, -1, -1):
            if input_id[i:i+end_id_len] == end_ids:
                labels = input_id.copy()
                labels[:i+end_id_len] = [-100] * (i+end_id_len)

                # 另外消除padding的影响  不需要算padding的损失
                for x in range(len(labels)):
                    if labels[x] == tokenizer.pad_token_id:
                        labels[x] = -100        
                target_ids.append(labels)
                break
            if i == 0:
                target_ids.append([-100] * input_len)
    return target_ids


def dpo_collate(batch, tokenizer, end_str, max_length):
    end_ids = tokenizer(end_str)['input_ids']
    end_id_len = len(end_ids)

    chosen_dialog_list = [item['chosen_dialog'] for item in batch]
    chosen_inputs = tokenizer(chosen_dialog_list, max_length=max_length, padding=True, truncation=True)
    chosen_input_ids = chosen_inputs['input_ids']
    chosen_target_ids = get_target_ids(chosen_input_ids, end_id_len, end_ids, tokenizer)
    chosen_inputs['labels'] = chosen_target_ids
    chosen_inputs = {k: torch.tensor(v) for k, v in chosen_inputs.items()}

    rejected_dialog_list = [item['rejected_dialog'] for item in batch]
    rejected_inputs = tokenizer(rejected_dialog_list, max_length=max_length, padding=True, truncation=True)
    rejected_input_ids = rejected_inputs['input_ids']
    rejected_target_ids = get_target_ids(rejected_input_ids, end_id_len, end_ids, tokenizer)
    rejected_inputs['labels'] = rejected_target_ids
    rejected_inputs = {k: torch.tensor(v) for k, v in rejected_inputs.items()}
    return chosen_inputs, rejected_inputs

