"""
@file   : 06-query_transform.py
@time   : 2025-07-11
"""
import fitz
import os
import numpy as np
import json
from openai import OpenAI


# Query Rewriting
def rewrite_query(original_query):
    # 对query进行改写
    # 重写查询以使其更具体和详细，以便更好地检索。
    system_prompt = "You are an AI assistant specialized in improving search queries. Your task is to rewrite user queries to be more specific, detailed, and likely to retrieve relevant information."

    user_prompt = f"""
        Rewrite the following query to make it more specific and detailed. Include relevant terms and concepts that might help in retrieving accurate information.

        Original query: {original_query}

        Rewritten query:
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


# Step-back Prompting
def generate_step_back_query(original_query):
    # 生成更通用的“后退”查询以检索更广泛的上下文。
    # “退后一步查询” 指的是生成一个更具通用性的查询语句，以便检索更广泛的背景信息或上下文。这种查询方式通常用于以下场景：当用户最初的查询过于具体或局限时，
    # 通过扩大搜索范围来获取更全面的信息，帮助理解问题的整体背景、相关领域的概况或潜在的关联因素。
    # Define the system prompt to guide the AI assistant's behavior
    system_prompt = "You are an AI assistant specialized in search strategies. Your task is to generate broader, more general versions of specific queries to retrieve relevant background information."

    # Define the user prompt with the original query to be generalized
    user_prompt = f"""
        Generate a broader, more general version of the following query that could help retrieve useful background information.

        Original query: {original_query}

        Step-back query:
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



# Sub-query Decomposition
def decompose_query(original_query, num_subqueries=4):
    # 将复杂查询分解为更简单的子查询。
    system_prompt = "You are an AI assistant specialized in breaking down complex questions. Your task is to decompose complex queries into simpler sub-questions that, when answered together, address the original query."

    # Define the user prompt with the original query to be decomposed
    user_prompt = f"""
        Break down the following complex query into {num_subqueries} simpler sub-queries. Each sub-query should focus on a different aspect of the original question.

        Original query: {original_query}

        Generate {num_subqueries} sub-queries, one per line, in this format:
        1. [First sub-query]
        2. [Second sub-query]
        And so on...
        """
    response = client.chat.completions.create(
        model="GLM-4-Flash-250414",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}],
        top_p=0.7,
        temperature=0.9
    )
    content = response.choices[0].message.content.strip()
    lines = content.split('\n')
    sub_queries = []
    for line in lines:
        if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, 10)):
            query = line.strip()
            query = query[query.find(".") + 1:].strip()
            sub_queries.append(query)
    return sub_queries


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



def process_document(pdf_path, chunk_size=1000, chunk_overlap=200, question_per_chunk=5):
    extracted_text = extract_pdf2text(pdf_path)
    text_chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)

    chunk_embeddings = create_embeddings(text_chunks)

    vector_store = SimpleVectorStore()
    for i, (chunk, embedding) in enumerate(zip(text_chunks, chunk_embeddings)):
        vector_store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={'index': i, 'source': pdf_path}
        )
    return vector_store


def transformed_search(query, vector_store, transformation_type, top_k=3):
    results = []
    if transformation_type == "rewrite":
        # Query rewriting
        transformed_query = rewrite_query(query)
        print(f"Rewritten query: {transformed_query}")

        # Create embedding for transformed query
        query_embedding = create_embeddings(transformed_query)

        # Search with rewritten query
        results = vector_store.similarity_search(query_embedding, k=top_k)

    elif transformation_type == "step_back":
        transformed_query = generate_step_back_query(query)
        print(f"Step-back query: {transformed_query}")

        # Create embedding for transformed query
        query_embedding = create_embeddings(transformed_query)

        # Search with step-back query
        results = vector_store.similarity_search(query_embedding, k=top_k)

    elif transformation_type == "decompose":
        sub_queries = decompose_query(query)
        print("Decomposed into sub-queries:")
        for i, sub_q in enumerate(sub_queries, 1):
            print(f"{i}. {sub_q}")

        # Create embeddings for all sub-queries
        sub_query_embeddings = create_embeddings(sub_queries)

        # Search with each sub-query and combine results
        all_results = []
        for i, embedding in enumerate(sub_query_embeddings):
            sub_results = vector_store.similarity_search(embedding, k=2)  # Get fewer results per sub-query
            all_results.extend(sub_results)

        # Remove duplicates (keep highest similarity score)
        seen_texts = {}
        for result in all_results:
            text = result["text"]
            if text not in seen_texts or result["similarity"] > seen_texts[text]["similarity"]:
                seen_texts[text] = result

        # Sort by similarity and take top_k
        results = sorted(seen_texts.values(), key=lambda x: x["similarity"], reverse=True)[:top_k]
    else:
        # Regular search without transformation
        query_embedding = create_embeddings(query)
        results = vector_store.similarity_search(query_embedding, k=top_k)

    return results



def generate_response(query, context):
    system_prompt = "You are an AI assistant that strictly answers based on the given context. If the answer cannot be derived directly from the provided context, respond with: 'I do not have enough information to answer that.'"

    user_prompt = f"""
            Context:
            {context}

            Question: {query}

            Please answer the question based only on the context provided above. Be concise and accurate.
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


def rag_with_query_transformation(pdf_path, query, transformation_type=None):
    vector_store = process_document(pdf_path)

    # Apply query transformation and search
    if transformation_type:
        # Perform search with transformed query
        results = transformed_search(query, vector_store, transformation_type)
    else:
        # Perform regular search without transformation
        query_embedding = create_embeddings(query)
        results = vector_store.similarity_search(query_embedding, k=3)

    # Combine context from search results
    context = "\n\n".join([f"PASSAGE {i + 1}:\n{result['text']}" for i, result in enumerate(results)])

    # Generate response based on the query and combined context
    response = generate_response(query, context)

    # Return the results including original query, transformation type, context, and response
    return {
        "original_query": query,
        "transformation_type": transformation_type,
        "context": context,
        "response": response
    }

if __name__ == '__main__':
    client = OpenAI(
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key="xxx"
    )

    # query = "我要去北京天安门玩，升旗时间是啥时候，另外周边有啥没事？ "
    # print(rewrite_query(query))
    # # "我想于2023年4月15日前往北京天安门广场观看升旗仪式，请问当天的具体升旗时间是什么时候？另外，我想了解天安门广场周边有哪些适合散步、拍照或休息的地点，以及附近有哪些著名的历史建筑或景点可以参观？请提供详细信息。"
    # print(generate_step_back_query(query))
    # # Step-back query: 我要去北京天安门参观，需要了解天安门的开放时间和附近有哪些旅游景点。
    # print(decompose_query(query))  # ['北京天安门的升旗时间是什么时候？', '北京天安门周边有哪些旅游景点？', '北京天安门周边有哪些餐饮选择？', '北京天安门周边有哪些娱乐活动？']

    data = json.load(open('./data/val.json', 'r', encoding='utf8'))
    query = data[0]['question']
    reference_answer = data[0]['ideal_answer']

    # pdf_path
    pdf_path = "data/AI_information.pdf"
    results = rag_with_query_transformation(pdf_path, query, transformation_type='rewrite')   # 指定几种改写方法: 'rewrite', 'step_back', 'decompose', 'original'
    print(results)




