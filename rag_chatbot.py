import os
from dotenv import load_dotenv
from embedding import Embedding
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
load_dotenv()

def detect_language(text: str) ->str:
    nepali_chars = sum(1 for ch in text if '\u0900' <=ch <= '\u097F')
    english_chars = sum(1 for ch in text if ch.isascii())

    if nepali_chars >0 and english_chars >0:
        return "mixed"
    elif nepali_chars > 0:
        return "Nepali"
    else:
        return "English"
    

class RAGChatBot:
    def __init__(self):
        print("Initializing RAG Chatbot")

        self.embeddings = Embedding().get_embedding_function()
        self.vectorstore = None
        self.temp_dir = "temp_uploads"
        
        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        
        self.llm = ChatGroq(
            model= "llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key = groq_key
        )
        print("✅ Ready!\n")

    def load_documents(self,chunks):
        valid_chunks = [
        chunk for chunk in chunks
        if chunk.page_content and chunk.page_content.strip()
    ]

        if not valid_chunks:
            print("⚠️ No valid chunks found!")
            return
        
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
        else:
            self.vectorstore.add_documents(chunks)
        print(f"✅ Loaded {len(chunks)} chunks into memory")

    def ask(self,question, k=3,fetch_k = 10, source_filter=None):

        docs_with_scores = self.vectorstore.similarity_search_with_score(
            question, k=fetch_k
        )

        if source_filter:
            docs_with_scores = [
                (doc, score) for doc, score in docs_with_scores
                if doc.metadata.get("source") == source_filter
            ]

        seen = set()
        docs = []
        for doc, score in docs_with_scores:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                docs.append(doc)
            if len(docs) == k:
                break
        
        if not docs:
            return {'answer': "I don't have information about that.", 'sources': {}}
        
        lang = detect_language(question)
        if lang == "Nepali":
            instruction = "Answer in Nepali"
        elif lang =="English":
            instruction = "Answer in English"
        else:
            instruction = "ANswer in a natural mix of Nepali and English"

        context = "\n\n".join([f"[Source {i+1}]\n{doc.page_content}" for i,doc in enumerate(docs)])
        prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided documents.
                                                 
Guidelines:
-{instruction}
- Give a direct and confident answer.
- Answer ONLY using information that is directly relevant to the question.
- IGNORE any unrelated or partially related information in the context.
- Do NOT include extra topics, categories, or examples unless they directly answer the question.
- If the context has NO relevant information, say exactly:
  "This information is not available in the uploaded documents."
- Do NOT make assumptions or add external knowledge.
- Keep the answer clean, focused, and strictly aligned with the question.

Context:
{context}

Question: {question}

Answer:"""

        
        answer = self.llm.invoke(prompt).content

        sources = list({doc.metadata.get("source", "unknown") for doc in docs})

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(docs)
        }

    def clear_all(self):
        self.vectorstore = None
        print("✅ Cleared vector store")