import faiss
import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Use a smaller, Streamlit-compatible model
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load FAISS index and chunk data
index = faiss.read_index("faiss_index/faiss.index")
with open("faiss_index/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

def embed_question(question):
    """Generate embeddings for a question using the model encoder."""
    inputs = tokenizer(question, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.encoder(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)  # Average token embeddings
    return embedding.squeeze().numpy().astype("float32")

def get_top_k_chunks(question, k=3):
    """Find the top k most relevant chunks to the given question."""
    try:
        question_embedding = embed_question(question)
        distances, indices = index.search(question_embedding.reshape(1, -1), k)
        return [chunks[i] for i in indices[0] if i < len(chunks)]
    except Exception as e:
        print(f"Error in get_top_k_chunks: {e}")
        return []

def build_prompt(question, top_chunks):
    """Create a prompt by appending context chunks before the question."""
    context = "\n\n".join(top_chunks)
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

def get_answer(prompt, model_name="google/flan-t5-small"):
    """Use flan-t5-small to generate an answer given a prompt."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
