import torch
from torch import nn
import torch.nn.functional as F


'''
在 MOE 中，单个专家的实现相比传统的 FeedForward 层多了一个 gate_proj，其作用是引入门控机制。具体来说：
gate_proj 的作用：
gate_proj 是一个线性层，用于生成门控值（gate values），这些值会与 up_proj 的输出进行逐元素相乘。
这种门控机制允许模型对不同的输入动态调整专家的行为。

计算流程：
输入 x 经过 gate_proj 和 up_proj，分别生成两个投影结果。
gate_proj(x) 的输出经过激活函数（如 silu），作为门控值。
门控值与 up_proj(x) 的输出逐元素相乘，形成加权的中间表示。
最后，结果通过 down_proj 投影回原始维度。
'''
class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
    def forward(self, x):
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Gating(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.topk = config.topk
        self.expert_num = config.expert_num
        self.gate = nn.Linear(self.hidden_size, self.expert_num)

    def forward(self, x):
        # print(x.size())   # torch.Size([4, 128, 512])    batch_size, max_len, hidden_size
        logits = self.gate(x)  # batch_size,
        # print(logits.size())  # torch.Size([4, 128, 4])   batch_size, max_len, gate_num

        logits_topk, indices = logits.topk(self.topk, dim=-1)  # 选择概率最大的两个专家，返回两个专家对每个token的概率
        # print(logits_topk.size())   # torch.Size([4, 128, 2])
        # print(indices.size())   # torch.Size([4, 128, 2])

        zeros = torch.full_like(logits, float("-inf"))  # 创建一个全为负无穷的矩阵，用于屏蔽其他专家的概率并重新归一化概率最大的两个专家
        # print(zeros.size())   # torch.Size([4, 128, 4])

        sparse_logits = zeros.scatter(dim=-1, index=indices, src=logits_topk)  # 将选择的两个专家的概率按指定索引填充
        sparse_logits = F.softmax(sparse_logits, dim=-1)  # 得到一个稀疏矩阵，选择的两个专家对每个token的概率和为1
        gate_logit = logits.view(-1, self.expert_num)
        # print(gate_logit.size())  # torch.Size([512, 4])

        # print(sparse_logits.size())  # torch.Size([4, 128, 4])
        # print(gate_logit.size())  # torch.Size([512, 4])
        # print(indices.size())   # torch.Size([4, 128, 2])
        return sparse_logits, indices, gate_logit


class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.expert_num)])
        self.gating = Gating(config)

    def forward(self, x):
        sparse_logits, indices, gate_logit = self.gating(x)
        # print(sparse_logits.size())  # torch.Size([4, 128, 4])
        # print(gate_logit.size())  # torch.Size([512, 4])
        # print(indices.size())   # torch.Size([4, 128, 2])

        x_flat = x.view(-1, x.shape[-1])  # (batch_size * seq_len, dim)
        sparse_logits_flat = sparse_logits.view(-1, sparse_logits.shape[-1])  # (batch_size * seq_len, export_num))   # 每个token的 多个专家的概率值。

        final_outputs = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):    # 遍历专家个数
            expert_mask = (indices == i).any(-1)  # (batch_size, seq_len)
            expert_mask_flat = expert_mask.view(-1)  # (batch_size * seq_len)
            if expert_mask_flat.any():
                expert_input = x_flat[expert_mask_flat]  # (seq_true, dim)   # 取出哪个token 在当前专家为True的向量
                export_output = expert(expert_input)  # (seq_true, dim)   # 输入专家  然后得到专家的输出。
                gate_scores = sparse_logits_flat[expert_mask_flat, i].unsqueeze(1)  # (seq_true) --> (seq_true, 1)
                weighted_output = export_output * gate_scores  # (seq_true, dim)
                final_outputs[expert_mask] += weighted_output
        return final_outputs, gate_logit


# 大模型中的MOE  替代的是之前的FeedForward这块  实际在计算中，可能不是每一块都经过moe  可能一层是标准FeedForward一层是MOE交替出现
"""
# 示例代码
hidden_states = self.post_attention_layernorm(hidden_states)   # 多头注意力最终的输出 batch_size, max_len, hidden_size
if self.layer_idx % 2 == 0:    # 偶数层 走标准FeedForward层
    hidden_states = self.mlp(hidden_states)
    gate_logit = None
else:    # 奇数层 走MOE层
    hidden_states, gate_logit = self.moe(hidden_states)
outputs = residual + hidden_states
return outputs, gate_logit
"""

class Config:
    expert_num: int = 4
    hidden_size: int = 512
    topk: int = 2
    intermediate_size: int = 2048
    mlp_bias: bool = False

config = Config()
moe = MoE(config)
hidden_states = torch.randn(4, 32, 512)  # batch_size, max_len, hidden_size
hidden_states, gate_logit = moe(hidden_states)
print(hidden_states.size())   # torch.Size([4, 128, 512])
print(gate_logit.size())   # torch.Size([512, 4])





