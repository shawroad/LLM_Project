"""
@file   : 03-context_enriched_retrieval.py
@time   : 2025-07-11
"""

# RAG中的上下文丰富检索
# 检索增强生成（RAG）通过从外部来源检索相关知识来增强AI响应。传统的检索方法返回孤立的文本块，这可能导致答案不完整。
# 为了解决这个问题，我们引入了上下文丰富检索，它确保检索到的信息包括相邻的块以获得更好的一致性。

# 步骤
# 1. 数据摄取：从PDF中提取文本。
# 2. 带有重叠上下文的分块：将文本拆分为重叠的块以保留上下文。
# 3. 嵌入创建：将文本块转换为数字表示。
# 4. 上下文感知检索：检索相关块及其邻居以获得更好的完整性。
# 5. 响应生成：使用语言模型根据检索到的上下文生成响应。


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



def chunk_text(text, n, overlap):
    # 文本切块
    # text: 文本,  n: 每块长短   overlap和之前重叠
    chunks = []
    for i in  range(0, len(text), n-overlap):
        chunks.append(text[i:i+n])
    return chunks


def create_embeddings(text):
    input_text = text if isinstance(text, list) else [text]
    response = client.embeddings.create(
        model='embedding-3',
        input=input_text
    )

    if isinstance(text, str):
        return response.data[0].embedding

    # Otherwise, return all embeddings as a list of vectors
    return [item.embedding for item in response.data]


def cosine_similarity(vec1, vec2):
    # 给定两个向量 计算相似度
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def context_enriched_search(query, text_chunks, embeddings, k=1, context_size=1):
    query_embedding = create_embeddings(query).data[0].embedding

    similarity_scores = []

    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding.embedding))
        similarity_scores.append((i, similarity_score))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)  # 排序

    top_index = similarity_scores[0][0]   # 最相关的那个document的索引

    start = max(0, top_index - context_size)   # 在最相关的索引 再往前推一点
    end = min(len(text_chunks), top_index + context_size + 1)   # 再往后推一点   说白了就是多带点上下文文本

    # Return the relevant chunk along with its neighboring context chunks
    return [text_chunks[i] for i in range(start, end)]


def generate_response(system_prompt, user_message):
    response = client.chat.completions.create(
        model="GLM-4-Flash-250414",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}],
        top_p=0.7,
        temperature=0.9
     )
    return response


if __name__ == '__main__':
    client = OpenAI(
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key="xxx"
    )

    text_extracted = extract_pdf2text('./data/AI_Information.pdf')
    text_chunks = chunk_text(text_extracted, 1000, 200)

    response = create_embeddings(text_chunks)

    data = json.load(open('./data/val.json', 'r', encoding='utf8'))

    query = data[0]['question']
    top_chunks = context_enriched_search(query, text_chunks, response.data, k=1, context_size=1)

    # print(query)
    # for i, chunk in enumerate(top_chunks):
    #     print('第{}段文本:{}'.format(i, chunk))

    system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"

    user_prompt = "\n".join([f"Context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in
                             enumerate(top_chunks)])  # top的所有文档
    user_prompt = f"{user_prompt}\nQuestion: {query}"  # 对应的query

    ai_response = generate_response(system_prompt, user_prompt)
    print(ai_response)


