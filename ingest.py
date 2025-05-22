import os
import faiss
import pickle
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def create_faiss_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

if __name__ == "__main__":
    file_path = "data/example.pdf"
    text = load_pdf(file_path)
    chunks = chunk_text(text)

    print(f"Loaded {len(chunks)} chunks.")

    embeddings = model.encode(chunks, show_progress_bar=True)
    index = create_faiss_index(embeddings)

    # Save index and chunks
    os.makedirs("vector_store", exist_ok=True)
    faiss.write_index(index, "vector_store/faiss.index")
    with open("vector_store/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("FAISS index and chunks saved.")
