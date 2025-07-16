"""
@file   : 04-contextual_chunk_headers_rag.py
@time   : 2025-07-11
"""
# 简单RAG中的上下文块标头（CCH）
# 检索-增强生成（RAG）通过在生成响应之前检索相关的外部知识来提高语言模型的事实准确性。然而，标准组块经常丢失重要的上下文，使得检索不太有效。
# 上下文块标头（CCH）通过在嵌入每个块之前为每个块添加高级上下文（如文档标题或部分标头）来增强RAG。这提高了检索质量并防止了断章取义的响应。
import fitz
import json
import numpy as np
from tqdm import tqdm
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


def generate_chunk_header(chunk):
    # 针对每个chunk生成一个简洁且信息丰富的标题
    system_prompt = "Generate a concise and informative title for the given text."
    response = client.chat.completions.create(
        model="GLM-4-Flash-250414",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk}],
        top_p=0.7,
        temperature=0.9
     )
    return response.choices[0].message.content.strip()


def chunk_text_with_headers(text, n, overlap):
    # 文本切块
    # text: 文本,  n: 每块长短   overlap和之前重叠
    chunks = []
    for i in  range(0, len(text), n-overlap):
        chunk = text[i:i + n]
        header = generate_chunk_header(chunk)   # 为每个chunk生成标题
        chunks.append({'header': header, 'text': chunk})
    return chunks


def create_embedding(text):
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


def semantic_search(query, chunks, k=5):
    query_embedding = create_embedding(query)

    similarities = []
    for chunk in chunks:
        sim_text = cosine_similarity(np.array(query_embedding), np.array(chunk['embedding']))
        sim_header = cosine_similarity(np.array(query_embedding), np.array(chunk['header_embedding']))
        avg_similarity = (sim_text + sim_header) / 2
        similarities.append((chunk, avg_similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in similarities[:k]]


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
    text_chunks = chunk_text_with_headers(text_extracted, 1000, 200)

    embeddings = []
    for chunk in tqdm(text_chunks, desc='Generating embeddings'):
        text_embedding = create_embedding(chunk['text'])
        header_embedding = create_embedding(chunk['header'])
        embeddings.append({
            'header': chunk['header'],
            'text': chunk['text'],
            'embedding': text_embedding,
            'header_embedding': header_embedding
        })

    data = json.load(open('./data/val.json', 'r', encoding='utf8'))

    query = data[0]['question']
    top_chunks = semantic_search(query, embeddings, k=2)

    # print(query)
    # for i, chunk in enumerate(top_chunks):
    #     print('第{}段文本:{}'.format(i, chunk))

    system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"

    user_prompt = "\n".join([f"Context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in
                             enumerate(top_chunks)])  # top的所有文档
    user_prompt = f"{user_prompt}\nQuestion: {query}"  # 对应的query

    ai_response = generate_response(system_prompt, user_prompt)
    print(ai_response)





