import openai
import fitz  # PyMuPDF
import faiss
import numpy as np
from typing import List

openai.api_key = "YOUR_OPENAI_API_KEY"

# 1. Extract text from PDF
def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    return " ".join(page.get_text() for page in doc)

# 2. Chunk the text
def chunk_text(text: str, size=500, overlap=50) -> List[str]:
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i+size])
    return chunks

# 3. Embed using OpenAI
def get_embedding(text: str) -> List[float]:
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response["data"][0]["embedding"]

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    return [get_embedding(chunk) for chunk in chunks]

# 4. Build FAISS index
def build_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index

# 5. Search for top-k chunks
def search_index(index, query, chunks, k=3):
    query_vec = np.array([get_embedding(query)]).astype("float32")
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]

# 6. Generate Answer using LLM
def generate_answer(context_chunks, question):
    context = "\n\n".join(context_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response["choices"][0]["message"]["content"]

# === RUNNING THE SYSTEM ===
pdf_text = extract_text_from_pdf("your_file.pdf")
chunks = chunk_text(pdf_text)
embeddings = embed_chunks(chunks)
index = build_index(embeddings)

query = "What is the document about?"
relevant_chunks = search_index(index, query, chunks)
answer = generate_answer(relevant_chunks, query)

print("\n💬 Answer:\n", answer)
