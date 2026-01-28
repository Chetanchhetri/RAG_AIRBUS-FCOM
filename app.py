import os
import fitz
import faiss
import requests
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

# ================= CONFIG =================
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"


PDF_PATH = "Airbus_dataset.pdf"
FAISS_INDEX_PATH = "faiss.index"
CHUNKS_PATH = "chunks.npy"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
TOP_K = 3

# ================= FASTAPI =================
app = FastAPI(title="RAG PDF Chat")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

# ================= PDF =================
def load_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    return " ".join(page.get_text() for page in doc)

# ================= CHUNKING =================
def chunk_text(text: str) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# ================= EMBEDDINGS =================
def embed_text(text: str) -> np.ndarray:
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60
    )

    if r.status_code != 200:
        raise HTTPException(500, f"Ollama embedding error: {r.text}")

    data = r.json()
    if "embedding" not in data:
        raise HTTPException(500, f"Invalid embedding response: {data}")

    emb = tf.convert_to_tensor([data["embedding"]], dtype=tf.float32).numpy()
    faiss.normalize_L2(emb)
    return emb

def embed_texts(texts: List[str]) -> np.ndarray:
    return np.array([embed_text(t)[0] for t in texts], dtype="float32")

# ================= VECTOR STORE =================
class VectorStore:
    def __init__(self):
        self.index = None
        self.texts = []

    def load_or_build(self):
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_PATH):
            print("⚡ Loading cached FAISS index")
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            self.texts = list(np.load(CHUNKS_PATH, allow_pickle=True))
            return

        print("📄 Loading PDF...")
        text = load_pdf_text(PDF_PATH)

        print("✂️ Chunking...")
        chunks = chunk_text(text)

        print("🔢 Creating embeddings...")
        embeddings = embed_texts(chunks)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.texts = chunks

        faiss.write_index(self.index, FAISS_INDEX_PATH)
        np.save(CHUNKS_PATH, chunks)

    def search(self, query: str) -> List[str]:
        q_emb = embed_text(query)
        _, idx = self.index.search(q_emb, TOP_K)
        return [self.texts[i] for i in idx[0]]

store = VectorStore()
store.load_or_build()

# ================= LLM =================
def generate_answer(context: str, question: str) -> str:
    prompt = f"""
Answer ONLY using the context.

Context:
{context}

Question:
{question}

Answer:
"""

    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    if r.status_code != 200:
        raise HTTPException(500, f"Ollama generate error: {r.text}")

    data = r.json()

    if "response" in data:
        return data["response"]

    if "error" in data:
        raise HTTPException(500, f"Ollama error: {data['error']}")

    raise HTTPException(500, f"Unexpected Ollama response: {data}")

# ================= API =================
class Query(BaseModel):
    question: str

@app.post("/chat")
def chat(query: Query):
    try:
        chunks = store.search(query.question)
        context = "\n\n".join(chunks)[:3000]
        answer = generate_answer(context, query.question)
        return {"answer": answer}
    except Exception as e:
        # ✅ ALWAYS return answer key
        return {"answer": f"Error: {str(e)}"}
