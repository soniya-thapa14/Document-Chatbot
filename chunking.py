from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        print("="*70)
        print("SPLITTING DOCUMENTS INTO CHUNKS")
        print("="*70)
        print(f"\n Chunk size: {self.chunk_size} characters")
        print(f'chunk overlap: {self.chunk_overlap} characters')
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            separators= ["\n\n","\n",". "," ",""]
        )
        chunks = splitter.split_documents(documents)
        valid_chunks = [
        chunk for chunk in chunks
        if chunk.page_content and chunk.page_content.strip()
    ]

        print(f"[INFO] {len(valid_chunks)} valid chunks from {len(chunks)} total")
        return valid_chunks

