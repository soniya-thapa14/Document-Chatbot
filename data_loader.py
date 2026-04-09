import os
import re
from langchain_community.document_loaders import(
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_core.documents import Document

supported_loaders = {
    ".pdf" :PyMuPDFLoader,
    ".docx" : Docx2txtLoader,
    ".txt" : lambda path: TextLoader(path, encoding="utf-8"),
}

def clean_text(text:str) -> str:
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{3,}','\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = text.strip()
    return text

def load_file(file_path: str) -> list[Document]:
    ext = os.path.splitext(file_path)[1].lower()

    if ext not in supported_loaders:
        print(f"skipping unsupportd files: {file_path}")
        return []
    
    LoaderClass = supported_loaders[ext]
    loader = LoaderClass(file_path)

    docs = loader.load()

    filename = os.path.basename(file_path)
    for doc in docs:
        doc.metadata["source"] =filename
        doc.page_content = clean_text(doc.page_content)
    return docs

def load_from_bytes(file_name:str ,file_bytes:bytes, temp_dir:str) -> list[Document]:
    temp_path = os.path.join(temp_dir, file_name)
    with open(temp_path,'wb') as f:
        f.write(file_bytes)
    docs = load_file(temp_path)

    if os.path.exists(temp_path):
        print(f"🗑️ Deleting temp file: {file_name}")
        os.remove(temp_path)
        print(f"✅ Deleted: {file_name}")

    return docs




