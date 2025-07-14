"""
@file   : inference.py
@time   : 2025-04-27
"""
import torch
from config import set_args
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(pretrain_model_path, lora_model_path):
    # 加载预训练模型和 LoRA 权重
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(pretrain_model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, lora_model_path)
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, user_input, max_length=1000, temperature=0.7, top_p=0.9):
    # 构造对话模板
    content = [{"role": "user", "content": user_input}]
    prompt = tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=True)
    # print(prompt)
    
    # 对输入进行分词
    input_max_length = 256
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=input_max_length)
    input_ids = inputs["input_ids"].cuda() if torch.cuda.is_available() else inputs["input_ids"]

    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )

    # 解码生成的文本
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # 提取回答部分
    assistant_start = response.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
    assistant_end = response.rfind("<|im_end|>")
    if assistant_start != -1 and assistant_end != -1:
        response = response[assistant_start:assistant_end].strip()
    return response

def generate_stream_response(model, tokenizer, user_input, max_length=1000, temperature=0.7, top_p=0.9):
    content = [{"role": "user", "content": user_input}]
    prompt = tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=True)
    # print(prompt)
    input_max_length = 256
    inputs = tokenizer(prompt, max_length=input_max_length, padding=True, truncation=True, return_tensors="pt")  
    generated_ids = inputs["input_ids"].cuda() if torch.cuda.is_available() else inputs["input_ids"]

    for _ in range(max_length):
        outputs = model.generate(
            input_ids=generated_ids,
            max_new_tokens=1,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )

        new_token_id = outputs[0, -1].unsqueeze(0).unsqueeze(0)
        generated_ids = torch.cat([generated_ids, new_token_id], dim=-1)

        new_token = tokenizer.decode(new_token_id[0], skip_special_tokens=True)
        print(new_token, end="", flush=True)
        if new_token_id.item() == tokenizer.eos_token_id:
            break




if __name__ == "__main__":
    # 配置路径
    args = set_args()
    lora_model_path = "./sft_model_output/sft_lora_model_best"

    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(args.pretrain_model, lora_model_path)
    if torch.cuda.is_available():
        model.cuda()

    # 用户输入
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("退出对话。")
            break
        # 生成回答
        print("Bot:")
        # response = generate_response(model, tokenizer, user_input)
        generate_stream_response(model, tokenizer, user_input)
        print()
