
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma

class ChromaStore:
    def __init__(self, embedding_function, ):
        self.embeddings = embedding_function
        self.vectorstore = None
        print(f"✅ In-memory vector store initialized")

    def build_vectorstore(self, chunks: List[Document]):
        self.vectorstore = Chroma.from_documents(
            documents= chunks,
            embedding= self.embeddings
        )
        print(f"✅ Vector store created with {len(chunks)} chunks")
