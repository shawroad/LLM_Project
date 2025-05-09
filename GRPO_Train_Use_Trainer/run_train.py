
import re
import torch
import datasets
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer



"""
torch='2.3.0+cu121'
transformers='4.51.3'
trl='0.17.0'
datasets='3.6.0'
"""


#从answer中提取计算结果；这些都是数学题，最终答案都是一个数字作为ground truth，数字和reason之间用####隔开的，所以用####做分割
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# 构造prompt，单独抽取answer
def get_gsm8k_questions(split = "train") -> Dataset:
    data = datasets.load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })    # type: ignore
    return data    # type: ignore


# Load and prep dataset：格式就是推理过程+最终结果
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    #把问题、答案、LLM的回复、从回复中抽取的结果都打印出来
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    #如果LLM的结果和训练样本的答案是一样的，说明回答正确，reward=2，奖励2分
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
    

#训练样本最终的结果是个数字，所以要求LLM最终输出的结果是数字，才能奖励0.5，reward=0.5
def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


#LLM的回复中如果有reasoning推理过程和answer结果标签，才符合既定的格式要求，这里reward=0.5
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


#同上，不过这里的正则检查没那么严格
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


if __name__ == "__main__":
    train_dataset = get_gsm8k_questions(split = "train")
    # print(train_dataset[10])
    '''
    {'question': 'A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate its hunger. Over three hundred years, it has consumed 847 people. Ships have been built larger over time, so each new ship has twice as many people as the last ship. How many people were on the ship the monster ate in the first hundred years?', 'answer': '121', 'prompt': [{'content': '\nRespond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n', 'role': 'system'}, {'content': 'A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate its hunger. Over three hundred years, it has consumed 847 people. Ships have been built larger over time, so each new ship has twice as many people as the last ship. How many people were on the ship the monster ate in the first hundred years?', 'role': 'user'}]}
    '''
    
    model_name = 'Qwen2.5-0.5B-Instruct'
    output_dir="outputs/Qwen2.5-0.5B-Instruct-GRPO"
    run_name="Qwen-1.5B-GRPO-gsm8k"

    # G= {num_processes} x {args.per_device_train_batch_size}   num_processes是GPU数量（单卡默认为1），per_device_train_batch_size是上面设置的变量，然后这个G必须被num_generations整除。这就很奇怪了
    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  #8k样本，这里有2k步梯度
        num_generations=16,
        max_prompt_length=256,
        max_completion_length=200,   #reasoning长度限制
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=False,
        vllm_gpu_memory_utilization=.3,
        vllm_device="cuda:0",
        report_to="none" #disabling Wandb.
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=None
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,  # 自定义的格式reward函数
            soft_format_reward_func,  # 自定义的格式reward函数
            strict_format_reward_func,  # 自定义的格式reward函数
            int_reward_func,   # 自定义的结果数字reward函数
            correctness_reward_func],   # 自定义的结果reward函数
        args=training_args,
        train_dataset=train_dataset,
        #peft_config=peft_config
    )
    trainer.train()
    trainer.save_model(output_dir)

