import openai
import faiss
import fitz
import numpy as np

openai.api_key = "YOUR_OPENAI_API_KEY"

# === STEP 1: PDF Loading & Chunking ===

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return " ".join(page.get_text() for page in doc)

def chunk_text(text, size=500, overlap=50):
    return [text[i:i+size] for i in range(0, len(text), size - overlap)]

# === STEP 2: Embedding ===

def get_embedding(text):
    resp = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return resp["data"][0]["embedding"]

def embed_chunks(chunks):
    return [get_embedding(chunk) for chunk in chunks]

# === STEP 3: Build FAISS Index ===

def build_index(embeds):
    dim = len(embeds[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeds).astype("float32"))
    return index

# === STEP 4: Multi-Query Rewriting ===

def generate_rewritten_queries(original_query, n=3):
    prompt = f"Generate {n} diverse but related queries for: '{original_query}'"
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    content = resp['choices'][0]['message']['content']
    return [q.strip("-• ").strip() for q in content.strip().split("\n") if q.strip()]

# === STEP 5: Search Top Chunks for Each Query ===

def search_top_chunks(queries, index, chunks, k=3):
    all_chunks = set()
    for q in queries:
        vec = np.array([get_embedding(q)]).astype("float32")
        _, indices = index.search(vec, k)
        for idx in indices[0]:
            all_chunks.add(chunks[idx])
    return list(all_chunks)

# === STEP 6: Generate Answer ===

def generate_answer(context_chunks, question):
    context = "\n\n".join(context_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return resp['choices'][0]['message']['content']

# === END-TO-END FLOW ===

pdf_text = extract_text_from_pdf("your_file.pdf")
chunks = chunk_text(pdf_text)
embeddings = embed_chunks(chunks)
index = build_index(embeddings)

query = "What are the key challenges mentioned in the document?"
rewritten_queries = generate_rewritten_queries(query)
print("🔁 Rewritten Queries:", rewritten_queries)

top_chunks = search_top_chunks(rewritten_queries, index, chunks)
answer = generate_answer(top_chunks, query)

print("\n💡 Answer:\n", answer)
