import streamlit as st
import os
import tempfile
import whisper
from ingest_dynamic import process_uploaded_pdf
from process_dataset import process_medical_dataset
from query import get_top_k_chunks, build_prompt, get_answer
from PIL import Image
import pytesseract

st.set_page_config(page_title="LLM Chat Assistant", layout="centered")
st.title("🧠 LLM-Powered Domain Chat Assistant")

# === Sidebar Settings ===
st.sidebar.header("⚙️ Settings")
use_ocr_qa = st.sidebar.checkbox("🖼️ Enable OCR Q&A", value=False)
use_voice_upload = st.sidebar.checkbox("🎙️ Enable Voice Input", value=True)

model_choice = st.sidebar.selectbox("🤖 Choose Model", ["flan-t5-base", "flan-t5-large", "google/flan-t5-small", "gpt2"], index=0)
domain = st.sidebar.selectbox("🩺 Select Domain", ["Generic", "Medical", "Legal"])
mode = st.sidebar.selectbox("📂 Data Source", ["Upload PDF", "Use Medical Dataset"])

# === PDF Upload Mode ===
if mode == "Upload PDF":
    st.markdown("---")
    st.subheader("📄 Upload a PDF Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file is not None:
        save_path = os.path.join("data", "uploaded.pdf")
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ PDF uploaded successfully!")
        with st.spinner("🔍 Embedding document..."):
            process_uploaded_pdf(save_path)
            st.success("✅ Document embedded and indexed!")

elif mode == "Use Medical Dataset":
    st.markdown("---")
    st.subheader("🏥 Load Medical Dataset")
    with st.spinner("Loading medical Q&A dataset..."):
        process_medical_dataset()
        st.success("✅ Medical dataset embedded and ready!")

# === OCR Question Answering ===
if use_ocr_qa:
    st.markdown("---")
    st.subheader("🖼️ OCR-Based Question Answering")

    uploaded_image = st.file_uploader("Upload an image (png/jpg/jpeg)", type=["png", "jpg", "jpeg"], key="img_upload")
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("Extracting text from image..."):
            ocr_text = pytesseract.image_to_string(image)
        st.text_area("📃 Extracted Text", ocr_text, height=200, key="ocr_text_img")
        image_question = st.text_input("❓ Question about the image content:", key="image_q")
        if st.button("💬 Submit Image Question"):
            if not image_question.strip():
                st.warning("❗ Please enter a question.")
            else:
                top_chunks = get_top_k_chunks(ocr_text)
                if not top_chunks:
                    st.warning("⚠️ No relevant context found. Answering without context.")
                    prompt = image_question
                else:
                    prompt = build_prompt(image_question, top_chunks, domain=domain)
                answer = get_answer(prompt, model_name=model_choice)
                st.markdown("**💡 Answer:**")
                st.write(answer)

    camera_image = st.camera_input("📷 Capture image with webcam", key="camera_input")
    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption="Captured Image", use_column_width=True)
        with st.spinner("Extracting text from captured image..."):
            ocr_text = pytesseract.image_to_string(image)
        st.text_area("📃 Extracted Text (camera)", ocr_text, height=200, key="ocr_text_cam")
        camera_question = st.text_input("❓ Question about captured image:", key="cam_q")
        if st.button("💬 Submit Camera Image Question"):
            if not camera_question.strip():
                st.warning("❗ Please enter a question.")
            else:
                top_chunks = get_top_k_chunks(ocr_text)
                if not top_chunks:
                    st.warning("⚠️ No relevant context found. Answering without context.")
                    prompt = camera_question
                else:
                    prompt = build_prompt(camera_question, top_chunks, domain=domain)
                answer = get_answer(prompt, model_name=model_choice)
                st.markdown("**💡 Answer:**")
                st.write(answer)

# === Voice Upload ===
if use_voice_upload:
    st.markdown("---")
    st.subheader("🎙️ Upload Audio for Transcription")
    audio_file = st.file_uploader("Upload audio (.wav or .mp3)", type=["wav", "mp3"], key="audio_upload")
    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            tmp_audio.write(audio_file.read())
            tmp_path = tmp_audio.name
        with st.spinner("🔊 Transcribing with Whisper..."):
            model = whisper.load_model("base")
            result = model.transcribe(tmp_path)
            transcribed_question = result["text"]
            st.success(f"📝 Transcribed: {transcribed_question}")
    else:
        transcribed_question = ""
else:
    transcribed_question = ""

# === Main QA Section ===
st.markdown("---")
st.subheader("💬 Ask a Question from the Document")
question = st.text_input("Type your question here:", value=transcribed_question)
if st.button("🚀 Submit Text Question") and question:
    with st.spinner("Thinking..."):
        top_chunks = get_top_k_chunks(question)
        if not top_chunks:
            st.warning("⚠️ No relevant context found. Answering without context.")
            prompt = question
        else:
            prompt = build_prompt(question, top_chunks, domain=domain)
        answer = get_answer(prompt, model_name=model_choice)
        st.markdown("**🔍 Retrieved Context:**")
        for i, chunk in enumerate(top_chunks):
            st.markdown(f"*Chunk {i+1}:* {chunk[:300]}...")
    st.markdown("---")
    st.subheader("💡 Answer")
    st.write(answer)
