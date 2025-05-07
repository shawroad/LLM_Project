import os
import torch
import functools
import pandas as pd
from config import set_args
from torch.utils.data import DataLoader
from data_helper import dpo_collate, DPODataset
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboard import SummaryWriter


def create_model():
    model = AutoModelForCausalLM.from_pretrained('../../Qwen2.5-0.5B-Instruct').to(device)
    tokenizer = AutoTokenizer.from_pretrained('../../Qwen2.5-0.5B-Instruct')
    return model, tokenizer


def dpo_prob_calc(target_ids,pi_logits,ref_logits):
    pi_probs=torch.log_softmax(pi_logits,dim=-1)      # softmax概率+log对数
    ref_probs=torch.log_softmax(ref_logits,dim=-1)
    
    ignore_mask= target_ids!=-100 # ignore token掩码
    indexes=target_ids*ignore_mask # 将-100变成0，以便后面gather可以运行

    pi_probs_of_target=torch.gather(pi_probs, dim=-1,index=indexes.unsqueeze(-1)).squeeze(-1) * ignore_mask # 取目标target token的概率，忽略-100 token
    ref_probs_of_target=torch.gather(ref_probs,dim=-1,index=indexes.unsqueeze(-1)).squeeze(-1) * ignore_mask    
    # print(pi_probs_of_target.size())   # torch.Size([2, 28])
    # print(ref_probs_of_target.size())   # torch.Size([2, 28])
    print(ignore_mask.sum(-1), ignore_mask.sum(-1))
    pi_final_prob=pi_probs_of_target.sum(-1)/ignore_mask.sum(-1)     # 求每一个样本的token prob均值
    ref_final_prob=ref_probs_of_target.sum(-1)/ignore_mask.sum(-1)
    return pi_final_prob, ref_final_prob
    

# DPO损失函数 https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py
def calc_dpo_loss(params):
    ## 两个模型的chosen输出
    chosen_target_ids=params['chosen_target_ids'][:,1:]
    # print(chosen_target_ids.size())  # torch.Size([2, 299])
    pi_chosen_logits=params['pi_chosen_logits'][:,:-1,:]
    # print(pi_chosen_logits.size())   # torch.Size([2, 299, 151936])
    ref_chosen_logits=params['ref_chosen_logits'][:,:-1,:]
    
    pi_chosen_prob, ref_chosen_prob=dpo_prob_calc(chosen_target_ids, pi_chosen_logits, ref_chosen_logits)
    # print(pi_chosen_prob.size())   # torch.Size([2])
    # print(ref_chosen_prob.size())   # torch.Size([2])
    
    ## 两个模型的reject输出
    reject_target_ids=params['reject_target_ids'][:,1:]
    pi_reject_logits=params['pi_reject_logits'][:,:-1,:]
    ref_reject_logits=params['ref_reject_logits'][:,:-1,:]
    pi_reject_prob,ref_reject_prob=dpo_prob_calc(reject_target_ids,pi_reject_logits,ref_reject_logits)
    
    # 计算DPO Loss
    pi_prob_diff=pi_chosen_prob-pi_reject_prob 
    ref_prob_diff=ref_chosen_prob-ref_reject_prob

    beta=0.01

    # 原始写法
    # loss=-torch.nn.functional.logsigmoid(beta*(pi_prob_diff-ref_prob_diff))

    # AI改法
    diff = beta * (pi_prob_diff - ref_prob_diff)
    diff = torch.clamp(diff, min=-50, max=50)  # 避免溢出
    loss = -torch.nn.functional.logsigmoid(diff)


    print(f"pi_prob_diff: {pi_prob_diff}, ref_prob_diff: {ref_prob_diff}")
    print(f"loss before mean: {loss}")
    
    return loss.mean()


def evaluate(test_dataloader):
    total_loss = 0
    model_pi.eval()
    for step, (chosen_inputs, rejected_inputs) in enumerate(test_dataloader):
        if torch.cuda.is_available():
            chosen_inputs = {k: v.cuda() for k, v in chosen_inputs.items()}
            rejected_inputs = {k: v.cuda() for k, v in rejected_inputs.items()}

        # 训练的模型
        pi_chosen_logits = model_pi(input_ids=chosen_inputs['input_ids'], attention_mask=chosen_inputs['attention_mask']).logits
        pi_rejected_logits = model_pi(input_ids=rejected_inputs['input_ids'], attention_mask=rejected_inputs['attention_mask']).logits
        # print(pi_chosen_logits.size())  # torch.Size([2, 75, 151936])
        # print(pi_rejected_logits.size())  # torch.Size([2, 170, 151936])

        # ref模型
        ref_chosen_logits = model_ref(input_ids=chosen_inputs['input_ids'], attention_mask=chosen_inputs['attention_mask']).logits
        ref_rejected_logits = model_ref(input_ids=rejected_inputs['input_ids'], attention_mask=rejected_inputs['attention_mask']).logits

        loss = calc_dpo_loss({
            'chosen_target_ids':chosen_inputs['labels'],
            'reject_target_ids':rejected_inputs['labels'],
            'pi_chosen_logits':pi_chosen_logits,
            'pi_reject_logits':pi_rejected_logits,
            'ref_chosen_logits':ref_chosen_logits,
            'ref_reject_logits':ref_rejected_logits
        })
        total_loss += loss.item()
    model_pi.train()
    return total_loss / len(test_dataloader)



if __name__ == '__main__':
    args = set_args()
    train_df = pd.read_parquet(args.train_data_path)
    test_df = pd.read_parquet(args.test_data_path)
    print("训练集:", train_df.shape)   # 训练集: (19862, 3)
    print("测试集:", test_df.shape)   # 测试集: (4996, 3)
    # print(train_df.head()) # prompt    chosen   rejected 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_pi, tokenizer = create_model()
    model_ref, _ = create_model()
    model_pi.train()
    model_ref.eval()

    train_dataset = DPODataset(train_df, tokenizer)
    test_dataset = DPODataset(test_df, tokenizer)

    collate_fn = functools.partial(dpo_collate,
                                   tokenizer=tokenizer,
                                   end_str="<|im_start|>assistant\n",
                                   max_length=args.max_len)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 优化器，只训练pi模型
    optimizer = torch.optim.AdamW(model_pi.parameters(), lr=1e-6)

    tb_write = SummaryWriter(log_dir=args.output_dir)
    global_step, tr_loss, logging_loss = 0, 0, 0
    eval_best_loss = float("inf")
    for epoch in range(args.num_epochs):
        for step, (chosen_inputs, rejected_inputs) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                chosen_inputs = {k: v.cuda() for k, v in chosen_inputs.items()}
                rejected_inputs = {k: v.cuda() for k, v in rejected_inputs.items()}
            
            assert not torch.isnan(chosen_inputs['input_ids']).any(), "NaN detected in input_ids"
            assert not torch.isinf(chosen_inputs['input_ids']).any(), "Inf detected in input_ids"

            # 训练的模型
            pi_chosen_logits = model_pi(input_ids=chosen_inputs['input_ids'], attention_mask=chosen_inputs['attention_mask']).logits
            pi_rejected_logits = model_pi(input_ids=rejected_inputs['input_ids'], attention_mask=rejected_inputs['attention_mask']).logits
            # print(pi_chosen_logits.size())  # torch.Size([2, 75, 151936])
            # print(pi_rejected_logits.size())  # torch.Size([2, 170, 151936])

            # ref模型
            ref_chosen_logits = model_ref(input_ids=chosen_inputs['input_ids'], attention_mask=chosen_inputs['attention_mask']).logits
            ref_rejected_logits = model_ref(input_ids=rejected_inputs['input_ids'], attention_mask=rejected_inputs['attention_mask']).logits

            loss = calc_dpo_loss({
                'chosen_target_ids':chosen_inputs['labels'],
                'reject_target_ids':rejected_inputs['labels'],
                'pi_chosen_logits':pi_chosen_logits,
                'pi_reject_logits':pi_rejected_logits,
                'ref_chosen_logits':ref_chosen_logits,
                'ref_reject_logits':ref_rejected_logits
            })
            print("epoch:{}, step:{}, loss: {:.5f}".format(epoch, step, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_pi.parameters(), max_norm=0.1)
            optimizer.step()

            global_step += 1
            tr_loss += loss.item()
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                tb_write.add_scalar("train_loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss

            if global_step % 5000 == 0:
                eval_loss = evaluate(test_dataloader)
                if eval_loss < eval_best_loss:
                    eval_best_loss = eval_loss
                    model_pi.save_pretrained(os.path.join(args.output_dir, 'dpo_model_best'))

                model_pi.save_pretrained(
                    os.path.join(args.output_dir, 'dpo_model_{}_step_{}'.format(epoch, global_step)))
                with open(os.path.join(args.output_dir, 'logs.txt'), 'a', encoding='utf8') as f:
                    f.write(f'epoch:{epoch}, step:{global_step}, eval_loss:{eval_loss}')