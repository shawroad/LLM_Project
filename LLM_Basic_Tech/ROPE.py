import torch


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    # print(xq.size())    # torch.Size([4, 299, 8, 64])
    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    # print(xq_r.size())   # torch.Size([4, 299, 8, 32])   # batch_size, max_len, head_nums, head_dim
    # print(xq_i.size())   # torch.Size([4, 299, 8, 32])

    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    # print(freqs_cos.size())   # torch.Size([299, 32])
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    # print(freqs_cos.size())   # torch.Size([1, 299, 1, 32])

    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos

    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    # print(xq_out.size())   # torch.Size([4, 299, 8, 64])
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # 可以参考飞书: AI笔记
    # print(dim)  # 64   单个头的维度
    # print(end)  # 最大长度
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # print(freqs.size())   # torch.Size([32])   针对一个向量 的最长维度64算的每个值
    t = torch.arange(end, device=freqs.device)  # type: ignore
    # print(t)   # [0, 1, 2, 3..., 299]
    freqs = torch.outer(t, freqs).float()  # type: ignore   # 有多少个token就对应多少向量
    # print(freqs)
    # print(freqs.size())   # torch.Size([300, 32])
    freqs_cos = torch.cos(freqs)  # real part
    # print(freqs_cos.size())   # torch.Size([300, 32])
    freqs_sin = torch.sin(freqs)  # imaginary part
    # print(freqs_sin.size())   # torch.Size([300, 32])
    return freqs_cos, freqs_sin



if __name__ == '__main__':
    dim = 512   # 向量维度
    n_heads = 8   # 多头的个数
    max_seq_len = 300   # 最长输入序列
    freqs_cos, freqs_sin = precompute_freqs_cis(dim // n_heads, max_seq_len)   # 算出sin和cos的周期值
    # print(freqs_cos.size())    #
    # print(freqs_sin.size())
    seq_len = 299   # 实际输入的长度
    freqs_cos = freqs_cos[:seq_len]
    freqs_sin = freqs_sin[:seq_len]

    # 随机造出多头的QKV
    xq = torch.randn(size=(4, 299, 8, 64))   # batch_size, max_len, head_num, head_dim
    xk = torch.randn(size=(4, 299, 8, 64))
    xv = torch.randn(size=(4, 299, 8, 64))

    xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
    print(xq.size())   # torch.Size([4, 299, 8, 64])
    print(xk.size())   # torch.Size([4, 299, 8, 64])
    # print(xq.size())
    # print(xk.size())
    # 此时 xq和xk就旋转可以了

