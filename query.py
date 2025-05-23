import faiss
import pickle
import torch
from transformers import AutoTokenizer, AutoModel

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModel.from_pretrained("google/flan-t5-base")

# Load FAISS index and chunk data
index = faiss.read_index("faiss_index/faiss.index")
with open("faiss_index/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

def embed_question(question):
    """Generate embeddings for a question using the Hugging Face model."""
    inputs = tokenizer(question, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)  # Average token embeddings
    return embedding.squeeze().numpy()

def get_top_k_chunks(question, k=3):
    """Find the top k most relevant chunks to the given question."""
    try:
        question_embedding = embed_question(question)
        distances, indices = index.search(question_embedding.reshape(1, -1), k)
        return [chunks[i] for i in indices[0]]
    except Exception as e:
        print(f"Error in get_top_k_chunks: {e}")
        return []

def build_prompt(question, top_chunks):
    """Create a prompt by appending context chunks before the question."""
    context = "\n\n".join(top_chunks)
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

def get_answer(prompt):
    """Use flan-t5-base to generate an answer given a prompt."""
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
    outputs = model.generate(input_ids, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
