"""
@file   : 08-relevant_segment_extraction.py
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
    def __init__(self, dimension=256):
        self.dimension = dimension
        self.vectors = []
        self.documents = []
        self.metadata = []


    def add_documents(self, documents, vectors=None, metadata=None):
        if vectors is None:
            vectors = [None] * len(documents)

        if metadata is None:
            metadata = [{} for _ in range(len(documents))]

        for doc, vec, meta in zip(documents, vectors, metadata):
            self.documents.append(doc)
            self.vectors.append(vec)
            self.metadata.append(meta)

    def search(self, query_vector, top_k=5):
        if not self.vectors or not self.documents:
            return []

        query_array = np.array(query_vector)

        similarities = []
        for i, vector in enumerate(self.vectors):
            if vector is not None:
                similarity = np.dot(query_array, vector) / (np.linalg.norm(query_array) * np.linalg.norm(vector))
                similarities.append((i, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, score in similarities[:top_k]:
            results.append({
                "document": self.documents[i],
                'score': float(score),
                'metadata': self.metadata[i]
            })
        return results


def create_embeddings(texts):
    if not texts:
        return []  # Return an empty list if no texts are provided

    # Process in batches if the list is long
    batch_size = 100  # Adjust based on your API limits
    all_embeddings = []  # Initialize a list to store all embeddings
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]  # Get the current batch of texts

        response = client.embeddings.create(
            model='embedding-3',
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)  # Add the batch embeddings to the list
    return all_embeddings


def process_document(pdf_path, chunk_size=800):
    text = extract_pdf2text(pdf_path)
    chunks = chunk_text(text, chunk_size, 0)
    chunk_embeddings = create_embeddings(chunks)

    vector_store = SimpleVectorStore()
    metadata = [{"chunk_index": i, "source": pdf_path} for i in range(len(chunks))]

    vector_store.add_documents(chunks, chunk_embeddings, metadata)

    doc_info = {
        "chunks": chunks,
        "source": pdf_path,
    }
    return chunks, vector_store, doc_info


def calculate_chunk_values(query, chunks, vector_store, irrelevant_chunk_penalty=0.2):
    query_embedding = create_embeddings([query])[0]   # 取出query的编码向量

    num_chunks = len(chunks)
    results = vector_store.search(query_embedding, top_k=num_chunks)  # 检索top_k

    relevance_scores = {result["metadata"]["chunk_index"]: result["score"] for result in results}
    # relevance_scores: {10: 0.43, 11: 0.43}  # 索引:得分

    chunk_values = []
    for i in range(num_chunks):
        score = relevance_scores.get(i, 0.0)
        value = score - irrelevant_chunk_penalty   # 当前得分 减去 惩罚
        chunk_values.append(value)
    return chunk_values


def find_best_segments(chunk_values, max_segment_length=20, total_max_length=30, min_segment_value=0.2):
    # 使用最大和子数组算法的变体找到最佳段。
    best_segments = []
    segment_scores = []
    total_included_chunks = 0

    while total_included_chunks < total_max_length:
        best_score = min_segment_value  # Minimum threshold for a segment
        best_segment = None

        for start in range(len(chunk_values)):
            if any(start >= s[0] and start < s[1] for s in best_segments):
                continue

            for length in range(1, min(max_segment_length, len(chunk_values) - start) + 1):
                end = start + length

                # 如果结束位置已经在选定的段中，则跳过
                if any(end > s[0] and end <= s[1] for s in best_segments):
                    continue

                # 计算段值作为块值的总和
                segment_value = sum(chunk_values[start:end])

                # 如果这个更好，请更新最佳片段
                if segment_value > best_score:
                    best_score = segment_value
                    best_segment = (start, end)

        # If we found a good segment, add it
        if best_segment:
            best_segments.append(best_segment)
            segment_scores.append(best_score)
            total_included_chunks += best_segment[1] - best_segment[0]
            print(f"Found segment {best_segment} with score {best_score:.4f}")
        else:
            # No more good segments to find
            break

    # 按起始位置对片段进行排序以提高易读性
    best_segments = sorted(best_segments, key=lambda x: x[0])
    return best_segments, segment_scores


def reconstruct_segments(chunks, best_segments):
    # 基于块索引重建文本段。
    reconstructed_segments = []  # Initialize an empty list to store the reconstructed segments
    for start, end in best_segments:
        # Join the chunks in this segment to form the complete segment text
        segment_text = " ".join(chunks[start:end])
        # Append the segment text and its range to the reconstructed_segments list
        reconstructed_segments.append({
            "text": segment_text,
            "segment_range": (start, end),
        })
    return reconstructed_segments  # Return the list of reconstructed text segments


def format_segments_for_context(segments):
    # Format segments into a context string for the LLM.
    context = []  # Initialize an empty list to store the formatted context
    for i, segment in enumerate(segments):
        # Create a header for each segment with its index and chunk range
        segment_header = f"SEGMENT {i + 1} (Chunks {segment['segment_range'][0]}-{segment['segment_range'][1] - 1}):"
        context.append(segment_header)  # Add the segment header to the context list
        context.append(segment['text'])  # Add the segment text to the context list
        context.append("-" * 80)  # Add a separator line for readability
    # Join all elements in the context list with double newlines and return the result
    return "\n\n".join(context)


def generate_response(query, context, model="meta-llama/Llama-3.2-3B-Instruct"):
    system_prompt = """You are a helpful assistant that answers questions based on the provided context.
    The context consists of document segments that have been retrieved as relevant to the user's query.
    Use the information from these segments to provide a comprehensive and accurate answer.
    If the context doesn't contain relevant information to answer the question, say so clearly."""

    user_prompt = f"""
Context:
{context}

Question: {query}

Please provide a helpful answer based on the context provided.
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


def rag_with_rse(pdf_path, query, chunk_size=800, irrelevant_chunk_penalty=0.2):
    chunks, vector_store, doc_info = process_document(pdf_path, chunk_size)  # 分段编码向量

    chunk_values = calculate_chunk_values(query, chunks, vector_store, irrelevant_chunk_penalty) # 计算chunk_values

    # Find the best segments of text based on chunk values
    best_segments, scores = find_best_segments(
        chunk_values,
        max_segment_length=20,
        total_max_length=30,
        min_segment_value=0.2
    )

    # Reconstruct text segments from the best chunks
    segments = reconstruct_segments(chunks, best_segments)

    # Format the segments into a context string for the language model
    context = format_segments_for_context(segments)

    # Generate a response from the language model using the context
    response = generate_response(query, context)

    # Compile the result into a dictionary
    result = {
        "query": query,
        "segments": segments,
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

    # Run the RAG with Relevant Segment Extraction (RSE) method
    rse_result = rag_with_rse(pdf_path, query)
    print(rse_result)







