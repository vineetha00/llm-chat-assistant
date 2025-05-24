import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load FAISS index and chunks
with open("faiss_index/index.pkl", "rb") as f:
    chunks = pickle.load(f)
index = faiss.read_index("faiss_index/index.faiss")

# Load the locally saved model and tokenizer
LOCAL_MODEL_DIR = "local_model"
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_DIR)

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Embedding model (same as before)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_top_k_chunks(question, k=3):
    question_embedding = embedding_model.encode([question])
    distances, indices = index.search(question_embedding, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def build_prompt(question, top_chunks):
    context = "\n".join(top_chunks)
    return f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

def get_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
