"""
@file   : 02-semantic_chunking.py
@time   : 2025-07-11
"""
import fitz
import json
import numpy as np
from openai import OpenAI


def extract_pdf2text(pdf_path):
    # 给定pdf文件路径  将内容全部抽取出来
    mypdf = fitz.open(pdf_path)
    all_text = ''
    for page in mypdf:
        all_text += page.get_text('text') + ' '
    return all_text


def get_embedding(text):
    # 给定一个文本 获取向量
    response = client.embeddings.create(
        model='embedding-3',
        input=[text]
    )
    # CreateEmbeddingResponse(data=[Embedding(embedding=[-0.0037525818, -0....])],
    #                         model='embedding-3',
    #                         object='list',
    #                         usage=Usage(prompt_tokens=463, total_tokens=463, completion_tokens=0))
    temp = response.data[0].embedding
    return np.array(temp)


def cosine_similarity(vec1, vec2):
    # 给定两个向量 计算相似度
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def compute_breakpoints(similarities, method='percentile', threshold=90):
    if method == 'percentile':
        # 计算相似度分数的第X百分位数
        threshold_value = np.percentile(similarities, threshold)   # 求百分之多少的分位数  从小到大   90%的分位数
    elif method == 'standard_deviation':
        # 计算相似度得分的均值和均方差
        mean = np.mean(similarities)
        std_dev = np.std(similarities)
        threshold_value = mean - (threshold * std_dev)
    elif method == 'interquartile':
        q1, q3 = np.percentile(similarities, [25, 75])
        threshold_value = q1 - 1.5 * (q3 - q1)
    else:
        raise ValueError("Invalid method. Choose 'percentile', 'standard_deviation', or 'interquartile'.")

    return [i for i, sim in enumerate(similarities) if sim < threshold_value]


def split_into_chunks(sentences, breakpoints):
    chunks = []
    start = 0
    for bp in breakpoints:
        chunks.append('. '.join(sentences[start: bp+1]) + '.')
        start = bp + 1
    chunks.append('. '.join(sentences[start:]))
    return chunks


if __name__ == '__main__':
    text_extracted = extract_pdf2text('./data/AI_Information.pdf')

    client = OpenAI(
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key="xxx"
    )

    sentences = text_extracted.split('. ')
    embeddings = [get_embedding(sentence) for sentence in sentences]   # 按句子打成向量

    similarities = [cosine_similarity(embeddings[i], embeddings[i+1]) for i in range(len(embeddings)-1)]  # 计算相邻之间的相似度分数

    breakpoints = compute_breakpoints(similarities, method="percentile", threshold=90)
    # print(breakpoints)
    text_chunks = split_into_chunks(sentences, breakpoints)
    print(text_chunks)


