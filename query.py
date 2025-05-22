from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import faiss
import numpy as np
import pickle

# Load model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load FAISS index and chunks
index = faiss.read_index("faiss_index/faiss.index")
with open("faiss_index/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Preprocess question
def embed_question(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.encoder(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy().astype("float32")

# Retrieve top-k chunks
def get_top_k_chunks(question, k=3):
    if not question.strip():
        return []
    query_vector = embed_question(question)
    D, I = index.search(query_vector, k)
    return [chunks[i] for i in I[0] if i < len(chunks)]

# Build prompt from context
def build_prompt(question, top_chunks):
    context = "\n".join(top_chunks)
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    return prompt

# Generate answer
def get_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
