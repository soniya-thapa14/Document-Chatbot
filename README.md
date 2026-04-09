📄 Document Chatbot (RAG)

A RAG-based document chatbot that allows users to upload files and ask questions based on their content.
Supports English & Nepali with source-backed answers.

🚀 Features
📂 Upload PDF, DOCX, TXT
🌐 Multilingual (English 🇬🇧 + Nepali 🇳🇵)
💬 Chat-based interface
📄 Answers with source references
🔍 Document-specific filtering
🗑️ Clear chat & documents

🛠️ Tech Stack

Streamlit
LangChain
ChromaDB
Sentence Transformers
Groq (LLaMA 3.3 70B)

⚙️ Run Locally

git clone https://github.com/soniya-thapa14/Document-Chatbot.git
cd document-chatbot
pip install -r requirements.txt

Create .env:

GROQ_API_KEY=your_api_key

streamlit run app.py
