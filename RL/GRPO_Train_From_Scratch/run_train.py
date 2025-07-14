from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from copy import deepcopy
from datasets import load_dataset
import os
import re
def extract_answer(text):
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def mark_num(text):
    reward = 0
    if text.count("<think>\n") == 1:
        reward += 0.125
        
    if text.count("</think>\n") == 1:
        reward += 0.125
        
    if text.count("<answer>\n") == 1:
        reward += 0.125
        
    if text.count("</answer>\n") == 1:
        reward += 0.125
    return reward


# 生成答案是否正确的奖励
def correctness_reward(prompts, responses, answers):
    
    extracted_responses = [extract_answer(r) for r in responses]
    print(f"问题:\n{prompts[0]}", f"\n答案:\n{answers[0]}", f"\n模型输出:\n{responses[0]}", f"\n提取后的答案:\n{extracted_responses[0]}")
    return [2.0 if response == str(ans) else 0.0 for response, ans in zip(extracted_responses, answers)]


# 生成答案是否是数字的奖励（单纯依赖结果是否正确进行奖励，条件很苛刻，会导致奖励比较稀疏，模型难以收敛，所以加上答案是否是数字的奖励，虽然答案错误，但是至少生成的是数字（对于数学问题），也要给予适当奖励）
def digit_reward(prompts, responses, answers):
    extracted_responses = [extract_answer(r) for r in responses]
    return [0.5 if response.isdigit() else 0.0 for response in extracted_responses]


# 格式奖励
def hard_format_reward(prompts, responses, answers):
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    matches = [re.match(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]


# 标记奖励（改善格式奖励稀疏问题）
def mark_reward(prompts, responses, answers):
    return [mark_num(response) for response in responses]


class GSM8KDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.answer = self.data['answer_only']
        self.question = self.data['question_zh-cn']
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        answer = self.answer[index]
        prompt = self.question[index]
        return {'prompt': prompt, 'answer': answer}


@dataclass
class Samples:
    prompt_response_ids: torch.Tensor
    response_ids: torch.Tensor
    prompt: Any
    answer: Any
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    response_length: int


class GRPOArguments:
    output_dir = './output'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.000001
    save_steps = 100
    epoch = 3
    num_generations = 4 # 组内样本数   一个输入 生成多少个样本
    max_prompt_length = 256 # 最大输入长度
    max_generate_length = 256 # 最大输出长度
    reward_weights : List[float] = None # 奖励的权重（多个奖励函数）
    beta = 0.0 # KL散度的系数，为0则忽略KL散度，即不使用参考模型
    clip_eps = 0.2
    gradient_accumulation_steps = 2 # 梯度累加
    num_iterations = 1 # 采样一次样本训练模型轮数
    batch_size = 1

class GRPOTrainer:
    def __init__(self,
        model = None,
        reward_funcs: Union[List[str], List[Callable]] = None,
        args = None,
        train_dataset: Optional[Union[Dataset]] = None,
        eval_dataset: Optional[Union[Dataset]] = None,
        tokenizer = None,
        reward_tokenizers = None):

        self.args = args

        # 加载模型
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model)
        self.model = model.to(self.args.device)
        
        # 是否使用参考模型
        self.ref_model = None
        if self.args.beta != 0.0:
            self.ref_model = deepcopy(model)
            self.ref_model.eval()
    
        
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        
        self.tokenizer = self.get_tokenizer(tokenizer)
        
        
        if isinstance(reward_funcs, str):
            reward_funcs = [reward_funcs]
        
        # 奖励可以是奖励模型  也可以是按规则打的分数
        for i, reward_func in enumerate(reward_funcs):
            # 如果奖励函数为字符串，表示使用的是奖励模型，则加载模型
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1).to(self.args.device)
        
        self.reward_funcs = reward_funcs
        
        if reward_tokenizers is None:
            reward_tokenizers = [None] * len(reward_funcs)
            
        elif isinstance(reward_tokenizers, str):
            reward_tokenizers = [reward_tokenizers]
            
        else:
            if len(reward_tokenizers) != len(reward_funcs):
                raise ValueError("Length of reward_tokenizers must be equal to the number of reward_funcs.")
            
        for i, (reward_tokenizer, reward_func) in enumerate(zip(reward_tokenizers, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_tokenizer is None:
                    reward_tokenizer = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_tokenizer.pad_token_id is None:
                    reward_tokenizer.pad_token = reward_tokenizer.eos_token
                
                reward_func.config.pad_token_id = reward_tokenizer.pad_token_id
                reward_tokenizers[i] = reward_tokenizer

        self.reward_tokenizers = reward_tokenizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # 缓存已经生成的数据的一个批次的数据，可供模型多次训练迭代，无需重新生成
        self.input_buffer = [None] * self.args.gradient_accumulation_steps
        
        # 模型更新的次数
        self.update_steps = 0 
    def get_tokenizer(self, tokenizer):
        tokenizer.padding_side = "left"
        return tokenizer
    
    # 生成样本，以组为单位
    def generate_samples(self, inputs):
        samples_list = []
        self.model.eval()
        prompts = [prompt for prompt in inputs['prompt']]
        answers = [None] * len(prompts)
        
        if 'answer' in inputs:
            answers = [answer for answer in inputs['answer']]
        
        max_length = self.args.max_generate_length + self.args.max_prompt_length
        for prompt, answer in zip(prompts, answers):
            # 应用聊天模板，加入系统提示词
            input_text = self.tokenizer.apply_chat_template([{"role": "system", 'content': SYSTEM_PROMPT}, {"role": "user", 'content': prompt}], add_generation_prompt=True, tokenize=False)
            
            # 生成一个group的输入数据
            # 针对一个输入  复制多个   为了生成多个输出
            inputs = self.tokenizer([input_text] * self.args.num_generations, padding='max_length', max_length=self.args.max_prompt_length, truncation=True, return_tensors='pt')
            prompt_ids = inputs['input_ids']
            with torch.no_grad():
                prompt_response_ids = self.model.generate(**inputs.to(self.args.device), 
                                    max_new_tokens = self.args.max_generate_length,
                                    temperature=0.9,
                                    top_p = 1,
                                    top_k = 50)
                
            if prompt_response_ids.size(1) >= max_length:
                prompt_response_ids = prompt_response_ids[:, :max_length]
            else:
                prompt_response_ids = torch.cat([prompt_response_ids, torch.full((prompt_response_ids.size(0), max_length - prompt_response_ids.size(1)), fill_value=self.tokenizer.pad_token_id, device=prompt_response_ids.device)], dim=1)
          
            attention_mask = (prompt_response_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)
            response_ids = prompt_response_ids[:, prompt_ids.size(1):]
            action_mask = (response_ids.ne(self.tokenizer.eos_token_id) & response_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)
        
            # 存储的是一个group的数据
            samples = Samples(
                prompt_response_ids=prompt_response_ids,   # 整体输出  包含不够 加了padding
                response_ids=response_ids,   # 只包含模型输出 不包含prompt的部分
                prompt = prompt,      # prompt文本
                answer = answer,    # 正确答案 
                attention_mask=attention_mask,  # prompt+回答   的mask矩阵
                action_mask=action_mask,    # 回答   的mask矩阵
                num_actions=action_mask.size(1),    # action的步数
                response_length=action_mask.float().sum(dim=-1)
            )
            samples_list.append(samples)

        return samples_list
    
    # 生成经验(优势、token的概率分布)
    def generate_experiences(self, inputs):
        self.model.eval()
        samples_list = self.generate_samples(inputs)
        # print(samples_list)   # 根据输入
        
        batch_prompt_response_ids = []
        batch_attention_mask = []
        batch_action_mask = []
        batch_advantages = []
        batch_old_action_log_probs = []
        batch_ref_action_log_probs = []
        
        for samples in samples_list:
            prompt_response_ids = samples.prompt_response_ids # shape: (num_generations, seq_len)
            # print(prompt_response_ids.size())   # torch.Size([4, 512])

            response_ids = samples.response_ids # shape: (num_generations, seq_len)
            # print(response_ids.size())   # torch.Size([4, 256])

            answer = samples.answer
            attention_mask = samples.attention_mask # shape: (num_generations, seq_len)
            action_mask = samples.action_mask # shape: (num_generations, seq_len)
            num_actions = samples.num_actions
            prompt = samples.prompt
            batch_prompt_response_ids.append(prompt_response_ids)
            batch_attention_mask.append(attention_mask)
            batch_action_mask.append(action_mask)
            
            with torch.no_grad():
                # 计算策略模型输出token的概率
                old_action_log_probs = self.get_action_log_probs(self.model, prompt_response_ids, attention_mask, num_actions)
                # print(old_action_log_probs.size())   # torch.Size([4, 256])
                batch_old_action_log_probs.append(old_action_log_probs)
                
                # 是否使用参考模型
                if self.ref_model:
                    #计算参考模型输出token的概率
                    ref_action_log_probs = self.get_action_log_probs(self.ref_model, prompt_response_ids, attention_mask, num_actions)
                    batch_ref_action_log_probs.append(ref_action_log_probs)
                
                # 存储各个奖励函数在一个group内各个响应的奖励
                rewards_per_func = torch.zeros(len(self.reward_funcs), self.args.num_generations, device=self.args.device)
                
                # 将输出转换成文本
                response_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                prompt_texts = [prompt] * len(response_texts)
                prompt_response_texts = [prompt + response for prompt, response in zip(prompt_texts, response_texts)]
                
                for i, (reward_func, reward_tokenizer) in enumerate(
                    zip(self.reward_funcs, self.reward_tokenizers)
                ):
                    if isinstance(reward_func, PreTrainedModel):
                        with torch.inference_mode():
                            reward_model_inputs = reward_tokenizer(prompt_response_texts, return_tensors="pt", padding=True)
                            rewards_per_func[i] = reward_func(**reward_model_inputs.to(self.args.device)).logits.squeeze(-1)
                    
                    else:
                        answers = [answer] * len(prompt_texts)
                        output_reward_func = reward_func(prompts=prompt_texts, responses=response_texts, answers=answers)
                        output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                        rewards_per_func[i] = torch.tensor(output_reward_func, dtype=torch.float32, device=self.args.device)
                # print(rewards_per_func)   # 多个输出 都打一个分数  torch.Size([4, 4]) 前面的4代表四个生成结果 后面4代表四个打分函数

                if not self.args.reward_weights:
                    self.args.reward_weights = [1.0] * len(self.reward_funcs)
                if len(self.args.reward_weights) != len(self.reward_funcs):
                    raise ValueError("The number of reward weights must be equal to the number of reward functions.")

                # 乘以各个奖励函数的权重
                rewards = rewards_per_func * torch.tensor(self.args.reward_weights, dtype=torch.float32, device=rewards_per_func.device).unsqueeze(1)
                
                # rewards: [num_funcs, num_generations]
                rewards = rewards.sum(dim=0) # shape: [num_generations]  torch.Size([4])
                print(f'rewards: {rewards}')
                mean_group_rewards = rewards.mean()
                std_group_rewards = rewards.std()
                
                # GRPO的优势是句子粒度的，而非token粒度的    归一化  
                advantages = (rewards - mean_group_rewards) / (std_group_rewards + 1e-8) # shape: [num_generations]
                batch_advantages.append(advantages)
        
               
        return {
            "prompt_response_ids": torch.cat(batch_prompt_response_ids, dim=0),
            "attention_mask": torch.cat(batch_attention_mask, dim=0),
            "action_mask": torch.cat(batch_action_mask, dim=0),
            "old_action_log_probs": torch.cat(batch_old_action_log_probs, dim=0),
            "ref_action_log_probs": torch.cat(batch_ref_action_log_probs, dim=0) if self.ref_model else None,
            "advantages": torch.cat(batch_advantages, dim=0),
        }
    
    def compute_loss(self, model, inputs):
        prompt_response_ids = inputs['prompt_response_ids']
        attention_mask = inputs['attention_mask']
        action_mask = inputs['action_mask']
        num_actions = action_mask.size(1)
        action_log_probs = self.get_action_log_probs(model, prompt_response_ids, attention_mask, num_actions)
        # print(action_log_probs.size())  # torch.Size([4, 256])
        
        if self.args.beta != 0.0:
            ref_action_log_probs = inputs['ref_action_log_probs']
            log_ratio = ref_action_log_probs - action_log_probs 
            log_ratio = log_ratio * action_mask
            
            # k3: log_ratio.exp() - 1 - log_ratio
            k3 = log_ratio.exp() - 1 - log_ratio
        
        advantages = inputs['advantages']   # 一个prompt生成多个样本  打分 归一化后的值
        # print(advantages.size())  # torch.Size([4])
        
        old_action_log_probs = inputs['old_action_log_probs'] if self.args.num_iterations > 1 else action_log_probs.detach()

        coef_1 = torch.exp(action_log_probs - old_action_log_probs) # 重要性采样 shape: [batch_size * num_generations, num_actions]
        # print(coef_1.size())   # torch.Size([4, 256])

        coef_2 = torch.clamp(coef_1, 1 - self.args.clip_eps, 1 + self.args.clip_eps)   # 裁剪
        # print(coef_2.size())   # torch.Size([4, 256])

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)   # 一个序列中每个token的优势是一样的
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)

        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = per_token_loss * action_mask
        if self.args.beta != 0.0:
            per_token_loss = per_token_loss + self.args.beta * k3
        loss = per_token_loss.sum() / action_mask.sum()
        return loss


    def get_action_log_probs(self, model, input_ids, attention_mask, num_actions):
        
        # 计算策略模型输出token的概率
        output = model(input_ids, attention_mask=attention_mask)
        logits = output.logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        log_probs_labels = log_probs.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))
        action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
        return action_log_probs

    
    
    def train_step(self, model, inputs, optimizer, step):
        model.train()
        # scaler = torch.amp.GradScaler()
        # with torch.amp.autocast(device_type='cuda'):
        loss = self.compute_loss(model, inputs)
        loss = loss / self.args.gradient_accumulation_steps
        # loss = scaler.scale(loss)
        loss.backward()
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            
            optimizer.step()
            optimizer.zero_grad()
            # scaler.unscale_(optimizer)
            # scaler.step(optimizer)
            # scaler.update()
        
            writer.add_scalar("grpo_loss", loss.item(), self.update_steps)
            print(f"step: {self.update_steps}/{self.global_steps}  grpo_loss: {loss.item():.8f}")
        torch.cuda.empty_cache()

    def train(self):
        self.global_steps = self.args.num_iterations * self.args.epoch * len(self.train_dataset) // (self.args.batch_size * self.args.gradient_accumulation_steps)

        for _ in range(self.args.epoch):
            dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
            for idx, batch in enumerate(dataloader):
                # print(batch)   # 是因为batch_size=1
                # {'prompt': ['安杰利有 90 种各种各样的糖果。三分之一的糖果是棒棒糖，其余的是拐杖糖。然后，她将棒棒糖平均分给男孩，每个男孩收到 3 个。然后，她平均分给女孩，每个女孩收到 2 个棒棒糖。总共给了多少个男孩和女孩？'], 
                #  'answer': tensor([40])}
                
                inputs = self.generate_experiences(batch)
                self.input_buffer[idx % self.args.gradient_accumulation_steps] = inputs
                if (idx + 1) % self.args.gradient_accumulation_steps == 0:
                    # 生成一次经验  迭代多次
                    for _ in range(self.args.num_iterations):
                        for step, inputs in enumerate(self.input_buffer):
                            self.train_step(self.model, inputs, self.optimizer, step)
                        
                        self.update_steps += 1
                        if self.update_steps % self.args.save_steps == 0:
                            self.model.save_pretrained(self.args.output_dir + f'/checkpoint_{self.update_steps}')
                            self.tokenizer.save_pretrained(self.args.output_dir + f'/checkpoint_{self.update_steps}')
                        
                del inputs
    def save_model(self):
        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)           

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_parquet("./train-00000-of-00001.parquet")  # huggface: swulling/gsm8k_chinese

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    SYSTEM_PROMPT = """
按照如下格式回答问题：
<think>
你的思考过程
</think>
<answer>
你的回答
</answer>
"""
    
    args = GRPOArguments()
    
    writer = SummaryWriter('./runs')
    # 策略模型
    tokenizer = AutoTokenizer.from_pretrained('../Qwen2.5-1.5B-Instruct')
    model = AutoModelForCausalLM.from_pretrained('../Qwen2.5-1.5B-Instruct')
    # 奖励函数
    # reward_model = '/home/user/Downloads/reward-model-deberta-v3-large-v2'
    # reward_tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/reward-model-deberta-v3-large-v2')
    
    
    prompts_dataset = GSM8KDataset(df, tokenizer)
    # print(prompts_dataset[10])
    # {'prompt': '深海怪物每隔百年就会从海里升起一次，以船上为食，充饥。三百年来，已经消耗了847人。随着时间的推移，船舶变得越来越大，因此每艘新船的人数是上一艘船的两倍。怪物在最初的一百年里吃掉了船上有多少人？', 'answer': 121}

  
    trainer = GRPOTrainer(model=model,
                          reward_funcs = [correctness_reward, digit_reward, hard_format_reward, mark_reward],
                          args=args,
                          train_dataset=prompts_dataset,
                          tokenizer=tokenizer)
    trainer.train()
    trainer.save_model()
    