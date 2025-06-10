import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == '__main__':
    pretrain_model_path = '../Qwen2.5-7B-Instruct'
    lora_model_path = 'sft_lora_model_1_step_2800'
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(pretrain_model_path, torch_dtype=torch.bfloat16, device_map="auto")
    
    # 加载LoRA适配器
    model = PeftModel.from_pretrained(model, lora_model_path)
        
    # 合并权重并卸载适配器
    merged_model = model.merge_and_unload()
        
    # 保存合并后的模型
    save_path = './Qwen2.5-7B-Instruct-With-Qlora'
    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
        

