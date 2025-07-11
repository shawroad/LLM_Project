"""
@file   : 01-simple_rag.py
@time   : 2025-07-11
"""
# python -m pip install --upgrade pymupdf
import fitz
import json
import numpy as np
from openai import OpenAI


def extract_pdf2text(pdf_path):
    # 给定pdf文件路径  将内容全部抽取出来
    mypdf = fitz.open(pdf_path)
    all_text = ''
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]
        text = page.get_text('text')  # 取出内容
        all_text += text
    return all_text


def chunk_text(text, n, overlap):
    # 文本切块
    # text: 文本,  n: 每块长短   overlap和之前重叠
    chunks = []
    for i in  range(0, len(text), n-overlap):
        chunks.append(text[i:i+n])
    return chunks

def create_embedding(texts):
    # texts是一个列表  里面放很多文本 [text1, text2, ...]
    response = client.embeddings.create(
        model='embedding-3',
        input=texts
    )
    # CreateEmbeddingResponse(data=[Embedding(embedding=[-0.0037525818, -0....])],
    #                         model='embedding-3',
    #                         object='list',
    #                         usage=Usage(prompt_tokens=463, total_tokens=463, completion_tokens=0))
    return response


def cosine_similarity(vec1, vec2):
    # 给定两个向量 计算相似度
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def semantic_search(query, text_chunks, embeddings, k=5):
    # 给定一个query 从所有候选文档中 做语义检索
    query_embedding = create_embedding(query).data[0].embedding
    similarity_scores = []
    for i, chunk_embedding in enumerate(embeddings):
        similarity_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding.embedding))
        similarity_scores.append((i, similarity_score))

    # 排序
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [index  for index, _ in similarity_scores[:k]]
    return [text_chunks[index] for index in top_indices]


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
    text_extracted = extract_pdf2text('./data/AI_Information.pdf')
    text_chunks = chunk_text(text_extracted, 1000, 200)
    # print(text_chunks)
    # print("文本块个数:{}".format(len(text_chunks)))   # 文本块个数:42
    # print(text_chunks[0])

    client = OpenAI(
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key="xxx"   # 智普的api
    )

    # 将分段后的文本编码向量
    response = create_embedding(text_chunks)

    # 加载所有query
    data = json.load(open('./data/val.json', 'r', encoding='utf8'))

    # 取第一个query测试一下
    query = data[0]['question']
    top_chunks = semantic_search(query, text_chunks, response.data, k=2)   # 选取top2相关的文档
    print("query:{}".format(query))
    print("检索top文档:{}".format(top_chunks))

    system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"

    user_prompt = "\n".join([f"Context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])   # top的所有文档
    user_prompt = f"{user_prompt}\nQuestion: {query}"   # 对应的query

    ai_response = generate_response(system_prompt, user_prompt)
    print(ai_response)

