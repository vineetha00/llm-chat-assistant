# 🧠 LLM-Powered Domain Chat Assistant

This is a Streamlit-based chat assistant that answers user questions using a custom knowledge base from PDFs, datasets, and OCR-extracted text. It supports domain-specific prompting, FAISS-based retrieval, and optional audio transcription using OpenAI Whisper.

## 🚀 Features

- 📄 Upload any PDF and chat with its content
- 🩺 Built-in medical dataset support
- 🖼️ OCR support for image and camera-captured text
- 🎙️ Voice input using Whisper (via audio file upload)
- 🔍 FAISS-based semantic search over embedded chunks
- 🧠 Prompt templates for generic, medical, and legal domains
- 🎛️ Model selector: flan-t5, gpt2, and more
- 🌐 Streamlit-based frontend

## 🗂 Folder Structure

```
llm-chat-assistant/
├── app.py                  # Main Streamlit app
├── query.py                # Core RAG logic (FAISS + prompting)
├── ingest_dynamic.py       # PDF ingestion
├── process_dataset.py      # Dataset loader
├── transcribe_audio.py     # Whisper-based audio transcription
├── faiss_index/            # Stores index.pkl and chunks.pkl
├── data/                   # For uploaded PDFs
├── requirements.txt
└── README.md
```

## 📦 Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/llm-chat-assistant.git
cd llm-chat-assistant

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## ☁️ Deploy on Streamlit

- Push your project to GitHub
- Go to https://streamlit.io/cloud
- Choose your repo and set `app.py` as the entry point
- Done!

## 🧠 Future Ideas

- Live camera feed for OCR
- Text-to-speech output
- Auto-indexing for multiple PDFs
- More domains & better prompt tuning

---

Created with ❤️ by Vineetha