import streamlit as st
import os
from data_loader import load_from_bytes
from chunking import DocumentChunker
from rag_chatbot import RAGChatBot

def detect_language(text: str) -> str:
    nepali_chars = sum(1 for ch in text if '\u0900' <= ch <= '\u097F')
    english_chars = sum(1 for ch in text if ch.isascii())

    if nepali_chars > 0 and english_chars > 0:
        return "mixed"
    elif nepali_chars > 0:
        return "nepali"
    else:
        return "english"

st.set_page_config(page_title="Document Chatbot", layout="wide")

st.title("📄 Document Chatbot")
st.markdown("Upload documents and ask questions based on their content.")

# ---------------- INITIALIZATION ---------------- #

if "bot" not in st.session_state:
    st.session_state.bot = RAGChatBot()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

bot: RAGChatBot = st.session_state.bot

# ---------------- SIDEBAR ---------------- #

st.sidebar.header("Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Choose files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}"
)

# Show loaded files
if st.session_state.processed_files:
    st.sidebar.markdown("### 📂 Loaded Documents")
    for name in st.session_state.processed_files:
        st.sidebar.write(f"✅ {name}")

# ---------------- FILE PROCESSING ---------------- #

if uploaded_files:
    temp_docs = []
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)

    for file in uploaded_files:
        if file.name not in st.session_state.processed_files:
            file_bytes = file.read()
            docs = load_from_bytes(file.name, file_bytes, temp_dir)
            temp_docs.extend(docs)

            st.session_state.processed_files.add(file.name)
            st.sidebar.info(f"📄 Processing {file.name}...")

    # Only load NEW docs
    if temp_docs:
        chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
        chunks = chunker.chunk_documents(temp_docs)
        st.session_state.bot.load_documents(chunks)

# ---------------- CLEAR ALL (FIXED) ---------------- #

if st.sidebar.button("🗑️ Clear All"):
    st.session_state.bot.clear_all()
    st.session_state.processed_files = set()
    st.session_state.messages = []
    st.session_state.uploader_key += 1
    
    st.rerun()


if st.session_state.processed_files:
    selected_file = st.sidebar.selectbox(
        "🔍 Search in:",
        ["All Documents"] + list(st.session_state.processed_files)
    )
else:
    selected_file = "All Documents"

# ---------------- CHAT INPUT ---------------- #

user_input = st.chat_input("Type your question here...")

if user_input:
    lang = detect_language(user_input)

    if lang == "nepali":
        st.info("🇳🇵 Nepali detected")
    elif lang == "english":
        st.info("🇬🇧 English detected")
    else:
        st.info("🌐 Mixed language detected")

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    if bot.vectorstore is None:
        reply = "📄 Please upload a document first before asking questions."
        sources = []
    else:
        source_filter = None if selected_file == "All Documents" else selected_file

        with st.spinner("🤔 Thinking..."):
            result = bot.ask(user_input, k=3, source_filter=source_filter)

        reply = result["answer"]
        sources = result["sources"]

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "sources": sources
    })

# ---------------- DISPLAY CHAT ---------------- #

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            sources = message.get("sources", [])
            if sources:
                st.markdown("**📚 Retrieved Sources:**")
                for src in sources:
                    st.write(f"📄 {src}")