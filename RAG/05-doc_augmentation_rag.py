"""
@file   : 05-doc_augmentation_rag.py
@time   : 2025-07-11
"""
import re
import fitz
import json
import numpy as np
from tqdm import tqdm
from openai import OpenAI

# 通过问题生成使用文档增强来实现增强的RAG方法。通过为每个文本块生成相关问题，我们改进了检索过程，从而从语言模型中获得更好的响应。
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


def generate_questions(text_chunk, num_questions=5):
    # 生成可以从给定文本片段中回答的相关问题。
    # Define the system prompt to guide the AI's behavior
    system_prompt = "You are an expert at generating relevant questions from text. Create concise questions that can be answered using only the provided text. Focus on key information and concepts."

    # Define the user prompt with the text chunk and the number of questions to generate
    user_prompt = f"""
    Based on the following text, generate {num_questions} different questions that can be answered using only this text:
    {text_chunk}
    Format your response as a numbered list of questions only, with no additional text.
    """

    response = client.chat.completions.create(
        model="GLM-4-Flash-250414",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}],
        top_p=0.7,
        temperature=0.9
     )

    questions_text = response.choices[0].message.content.strip()
    questions = []
    # Extract questions using regex pattern matching
    for line in questions_text.split('\n'):
        # Remove numbering and clean up whitespace
        cleaned_line = re.sub(r'^\d+\.\s*', '', line.strip())
        if cleaned_line and cleaned_line.endswith('?'):
            questions.append(cleaned_line)
    return questions


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


def process_document(pdf_path, chunk_size=1000, chunk_overlap=200, question_per_chunk=5):
    extracted_text = extract_pdf2text(pdf_path)

    text_chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)

    vector_store = SimpleVectorStore()

    for i, chunk in enumerate(tqdm(text_chunks, desc='Processing Chunks')):
        chunk_embedding_response = create_embedding(chunk)
        chunk_embedding = chunk_embedding_response.data[0].embedding

        vector_store.add_item(text=chunk, embedding=chunk_embedding, metadata={"type": "chunk", "index": i})

        questions = generate_questions(chunk, num_questions=question_per_chunk)
        # print(questions)

        for j, question in enumerate(questions):
            question_embedding_response = create_embedding(question)
            question_embedding = question_embedding_response.data[0].embedding
            vector_store.add_item(text=question, embedding=question_embedding, metadata={"type": "question", "chunk_index": i, 'original_chunk': chunk})
    return text_chunks, vector_store


def semantic_search(query, vector_store, k=5):
    query_embedding_response = create_embedding(query)
    query_embedding = query_embedding_response.data[0].embedding

    results = vector_store.similarity_search(query_embedding, k=k)
    return results


def prepare_context(search_results):
    chunk_indices = set()
    context_chunks = []

    for result in search_results:
        if result['metadata']['type'] == 'chunk':
            chunk_indices.add(result['metadata']['index'])
            context_chunks.append(f"Chunk {result['metadata']['index']}:\n{result['text']}")

    for result in search_results:
        if result['metadata']['type'] == 'question':
            chunk_idx = result['metadata']['chunk_index']
            if chunk_idx not in chunk_indices:
                chunk_indices.add(chunk_idx)
                context_chunks.append(f"Chunk {chunk_idx} (referenced by question '{result['text']}'):\n{result['metadata']['original_chunk']}")
    full_context = '\n\n'.join(context_chunks)
    return full_context



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
    return response.choices[0].message.content


if __name__ == '__main__':
    client = OpenAI(
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key="xxx"
    )

    text_chunks, vector_store = process_document('./data/AI_Information.pdf', chunk_size=1000, chunk_overlap=200, question_per_chunk=3)

    # 加载所有query
    data = json.load(open('./data/val.json', 'r', encoding='utf8'))

    # 取第一个query测试一下
    query = data[0]['question']

    search_results = semantic_search(query, vector_store, k=5)


    context = prepare_context(search_results)
    response_text = generate_response(query, context)

    print("query:{}".format(query))
    print("response:{}".format(response_text))
