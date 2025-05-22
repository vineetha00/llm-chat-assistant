import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load FAISS index and chunks
index_path = "faiss_index/index.pkl"
chunks_path = "faiss_index/chunks.pkl"
model = SentenceTransformer("all-MiniLM-L6-v2")

with open(chunks_path, "rb") as f:
    chunks = pickle.load(f)

with open(index_path, "rb") as f:
    index = pickle.load(f)

def get_top_k_chunks(question, k=3, score_threshold=0.2):
    question_embedding = model.encode([question])[0]
    D, I = index.search(np.array([question_embedding]), k)
    top_chunks = []
    for i, score in zip(I[0], D[0]):
        if score > score_threshold and i != -1:
            chunk = chunks[i].strip()
            if len(chunk) > 100:
                top_chunks.append(chunk[:300])  # truncate long chunks
    return top_chunks

def build_prompt(question, top_chunks):
    context = "\n".join(top_chunks)
    return f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

def get_answer(prompt, model_name="flan-t5-base"):
    pipe = pipeline("text2text-generation", model=model_name)
    result = pipe(prompt, max_new_tokens=256)
    return result[0]["generated_text"]
