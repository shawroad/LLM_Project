"""
@file   : 07-reranker.py
@time   : 2025-07-14
"""
import re

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



class SimpleVectorStore:
    # 向量近似计算
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add_item(self, text, embedding, metadata=None):
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})

    def similarity_search(self, query_embedding, k=5):
        if not self.vectors:
            return []

        query_vector = np.array(query_embedding)

        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": score
            })
        return results


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


def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    extracted_text = extract_pdf2text(pdf_path)
    chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)

    chunk_embeddings = create_embeddings(chunks)

    store = SimpleVectorStore()

    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={"index": i, 'source': pdf_path}
        )
    return store


def rerank_with_llm(query, results, top_n=3):
    # 使用llm对query和每个doc进行打分
    scored_results = []

    system_prompt = """You are an expert at evaluating document relevance for search queries.
Your task is to rate documents on a scale from 0 to 10 based on how well they answer the given query.

Guidelines:
- Score 0-2: Document is completely irrelevant
- Score 3-5: Document has some relevant information but doesn't directly answer the query
- Score 6-8: Document is relevant and partially answers the query
- Score 9-10: Document is highly relevant and directly answers the query

You MUST respond with ONLY a single integer score between 0 and 10. Do not include ANY other text.
    """
    for i, result in enumerate(results):
        user_prompt = f"""Query:{query}
        
Document:
{result['text']}

Rate this document's relevance to the query on a scale from 0 to 10:"""
        response = client.chat.completions.create(
            model="GLM-4-Flash-250414",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}],
            top_p=0.7,
            temperature=0.9
        )
        score_text = response.choices[0].message.content.strip()
        score_match = re.search(r'\b(10|[0-9])\b', score_text)
        if score_match:
            score = float(score_match.group(1))
        else:
            # 如果分数提取失败，使用相似度分数作为后备
            score = result['similarity'] * 10

        scored_results.append({
            "text": result['text'],
            'metadata': result['metadata'],
            'similarity': result['similarity'],
            'relevance_score': score
        })

    reranked_results = sorted(scored_results, key=lambda x: x['relevance_score'], reverse=True)
    return reranked_results[:top_n]


def rerank_with_keywords(query, results, top_n=3):
    keywords = [word.lower() for word in query.split() if len(word) > 3]   # 抽取query中的关键字(这里的关键字长度要求>3)

    scored_results = []
    for result in results:
        document_text = result['text'].lower()

        base_score = result['similarity'] * 0.5

        keyword_score = 0
        for keyword in keywords:
            if keyword in document_text:
                keyword_score += 0.1

                first_position = document_text.find(keyword)
                if first_position < len(document_text) / 4:   # 出现的位置在0-1/4位置  更重要
                    keyword_score += 0.1

                frequency = document_text.count(keyword)   # 将频次加权到分数中
                keyword_score += min(0.05 * frequency, 0.2)

        first_score = base_score + keyword_score

        scored_results.append({
            "text": result['text'],
            'metadata': result['metadata'],
            'similarity': result['similarity'],
            'relevance_score': first_score
        })

    reranked_results = sorted(scored_results, key=lambda x: x['relevance_score'], reverse=True)
    return reranked_results[:top_n]



def generate_response(query, context):
    system_prompt = "You are a helpful AI assistant. Answer the user's question based only on the provided context. If you cannot find the answer in the context, state that you don't have enough information."

    user_prompt = f"""
            Context:
            {context}

            Question: {query}
            
            Please provide a comprehensive answer based only on the context above.
    """

    response = client.chat.completions.create(
        model="GLM-4-Flash-250414",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}],
        top_p=0.7,
        temperature=0.9
     )
    return response.choices[0].message.content.strip()



def rag_with_reranking(query, vector_store, reranking_method='llm', top_n=3):
    query_embedding = create_embeddings(query)

    initial_results = vector_store.similarity_search(query_embedding, k=10)  # 先根据向量检索出最相关的10个

    if reranking_method == 'llm':
        reranked_results = rerank_with_llm(query, initial_results, top_n=top_n)
    elif reranking_method == 'keywords':
        reranked_results = rerank_with_keywords(query, initial_results, top_n=top_n)
    else:
        reranked_results = initial_results[:top_n]  # 直接截断

    context = "\n\n===\n\n".join([result["text"] for result in reranked_results])

    response = generate_response(query, context)

    return {
        "query": query,
        "reranking_method": reranking_method,
        "initial_results": initial_results[:top_n],
        "reranked_results": reranked_results,
        "context": context,
        "response": response
    }


if __name__ == '__main__':
    client = OpenAI(
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key="xxx"
    )

    pdf_path = "data/AI_information.pdf"
    vector_store = process_document(pdf_path=pdf_path)

    query = "Does AI have the potential to transform the way we live and work?"


    standard_results = rag_with_reranking(query, vector_store, reranking_method="none")
    print('query:', query)
    print('response:', standard_results)

    standard_results = rag_with_reranking(query, vector_store, reranking_method="llm")
    print('query:', query)
    print('response:', standard_results)

    standard_results = rag_with_reranking(query, vector_store, reranking_method="keywords")
    print('query:', query)
    print('response:', standard_results)



