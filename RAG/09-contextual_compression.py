"""
@file   : 09-contextual_compression.py
@time   : 2025-07-15
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
            metadata={"index": i, "source": pdf_path}
        )
    return store



def compress_chunk(chunk, query, compression_type="selective"):
    # 通过仅保留与查询相关的部分来压缩检索到的块。
    if compression_type == "selective":
        system_prompt = """You are an expert at information filtering. 
        Your task is to analyze a document chunk and extract ONLY the sentences or paragraphs that are directly 
        relevant to the user's query. Remove all irrelevant content.

        Your output should:
        1. ONLY include text that helps answer the query
        2. Preserve the exact wording of relevant sentences (do not paraphrase)
        3. Maintain the original order of the text
        4. Include ALL relevant content, even if it seems redundant
        5. EXCLUDE any text that isn't relevant to the query

        Format your response as plain text with no additional comments."""

    elif compression_type == "summary":
        system_prompt = """You are an expert at summarization. 
        Your task is to create a concise summary of the provided chunk that focuses ONLY on 
        information relevant to the user's query.

        Your output should:
        1. Be brief but comprehensive regarding query-relevant information
        2. Focus exclusively on information related to the query
        3. Omit irrelevant details
        4. Be written in a neutral, factual tone

        Format your response as plain text with no additional comments."""

    else:  # extraction
        system_prompt = """You are an expert at information extraction.
        Your task is to extract ONLY the exact sentences from the document chunk that contain information relevant 
        to answering the user's query.

        Your output should:
        1. Include ONLY direct quotes of relevant sentences from the original text
        2. Preserve the original wording (do not modify the text)
        3. Include ONLY sentences that directly relate to the query
        4. Separate extracted sentences with newlines
        5. Do not add any commentary or additional text

        Format your response as plain text with no additional comments."""

    # Define the user prompt with the query and document chunk
    user_prompt = f"""
        Query: {query}

        Document Chunk:
        {chunk}

        Extract only the content relevant to answering this query.
    """


    response = client.chat.completions.create(
        model="GLM-4-Flash-250414",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}],
        top_p=0.7,
        temperature=0.9
     )

    # Extract the compressed chunk from the response
    compressed_chunk = response.choices[0].message.content.strip()

    # Calculate compression ratio
    original_length = len(chunk)
    compressed_length = len(compressed_chunk)
    compression_ratio = (original_length - compressed_length) / original_length * 100
    return compressed_chunk, compression_ratio


def batch_compress_chunks(chunks, query, compression_type="selective"):
    # 单独压缩多个块。
    results = []
    total_original_length = 0
    total_compressed_length = 0

    # Iterate over each chunk
    for i, chunk in enumerate(chunks):
        compressed_chunk, compression_ratio = compress_chunk(chunk, query, compression_type)
        results.append((compressed_chunk, compression_ratio))

        total_original_length += len(chunk)
        total_compressed_length += len(
            compressed_chunk)

    overall_ratio = (total_original_length - total_compressed_length) / total_original_length * 100
    print(overall_ratio)
    return results


def generate_response(query, context):
    system_prompt = """You are a helpful AI assistant. Answer the user's question based only on the provided context.
    If you cannot find the answer in the context, state that you don't have enough information."""

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
    return response.choices[0].message.content


def rag_with_compression(pdf_path, query, k=10, compression_type="selective"):
    vector_store = process_document(pdf_path)

    query_embedding = create_embeddings(query)

    results = vector_store.similarity_search(query_embedding, k=k)
    retrieved_chunks = [result["text"] for result in results]

    compressed_results = batch_compress_chunks(retrieved_chunks, query, compression_type)
    compressed_chunks = [result[0] for result in compressed_results]
    compression_ratios = [result[1] for result in compressed_results]

    # Filter out any empty compressed chunks
    filtered_chunks = [(chunk, ratio) for chunk, ratio in zip(compressed_chunks, compression_ratios) if chunk.strip()]

    if not filtered_chunks:
        print("Warning: All chunks were compressed to empty strings. Using original chunks.")
        filtered_chunks = [(chunk, 0.0) for chunk in retrieved_chunks]
    else:
        compressed_chunks, compression_ratios = zip(*filtered_chunks)

    context = "\n\n---\n\n".join(compressed_chunks)

    # Generate a response based on the compressed chunks
    print("Generating response based on compressed chunks...")
    response = generate_response(query, context)

    # Prepare the result dictionary
    result = {
        "query": query,
        "original_chunks": retrieved_chunks,
        "compressed_chunks": compressed_chunks,
        "compression_ratios": compression_ratios,
        "context_length_reduction": f"{sum(compression_ratios) / len(compression_ratios):.2f}%",
        "response": response
    }
    return result


if __name__ == '__main__':
    client = OpenAI(
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        api_key="xxx"
    )

    pdf_path = "data/AI_information.pdf"
    query = "Does AI have the potential to transform the way we live and work?"
    comp_type = 'selective'
    result = rag_with_compression(pdf_path, query, compression_type=comp_type)
    print(result)


