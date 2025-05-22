import pandas as pd
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_dataset_rows(file_path):
    df = pd.read_csv(file_path)
    # Concatenate question and answer
    chunks = [f"Q: {row['question']}\nA: {row['answer']}" for _, row in df.iterrows()]
    return chunks

def create_faiss_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def process_medical_dataset(file_path="data/medical_qa.csv"):
    print("Loading medical Q&A dataset...")
    chunks = chunk_dataset_rows(file_path)

    # Optional filtering + truncation for cleaner results
    chunks = [chunk[:300] for chunk in chunks if len(chunk) > 100]

    print("Embedding...")
    embeddings = model.encode(chunks, show_progress_bar=True)

    print("Saving FAISS index and chunks...")
    os.makedirs("vector_store", exist_ok=True)
    faiss.write_index(create_faiss_index(embeddings), "vector_store/faiss.index")
    with open("vector_store/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Done. Medical dataset embedded and indexed.")

# For standalone CLI use
if __name__ == "__main__":
    process_medical_dataset()
