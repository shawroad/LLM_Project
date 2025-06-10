import torch
import time
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

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


if __name__ == '__main__':
    pretrain_model_path = 'Qwen2.5-7B-Instruct-With-Qlora'
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(pretrain_model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    start_time = time.time()
    for i in range(100):
        res = generate_response(model, tokenizer, user_input='根据下列商品描述生成一个优质的微博广告文案。商品描述：热敷毛巾面膜罩冷敷蒸脸敷脸巾美容院用面部脸部蒸汽加热面罩眼巾')
        print(res)
    end_time = time.time()
    print((end_time - start_time) / 100)