"""Helper functions for Rafiqa Pre-Marital Health Assistant."""
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document


def load_pdf_file(data_path: str) -> List[Document]:
    """Load PDF files from a directory."""
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()
    return documents


def load_text_file(file_path: str) -> List[Document]:
    """Load a single .txt or .md file."""
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    return documents


def load_files_from_dir(data_path: str) -> List[Document]:
    """Load all supported files (pdf, txt, md) from a directory."""
    import os
    all_docs = []
    for fname in os.listdir(data_path):
        fpath = os.path.join(data_path, fname)
        ext = fname.lower().rsplit('.', 1)[-1] if '.' in fname else ''
        if ext == 'pdf':
            loader = PyPDFLoader(fpath)
            all_docs.extend(loader.load())
        elif ext in ('txt', 'md'):
            loader = TextLoader(fpath, encoding='utf-8')
            all_docs.extend(loader.load())
    return all_docs


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Filter documents to keep only essential metadata (source).
    Reduces overhead when storing in vector database.
    """
    minimal_docs = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src},
            )
        )
    return minimal_docs


def text_split(
    extracted_data: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """Split documents into smaller text chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


def download_hugging_face_embeddings():
    """Download and return HuggingFace embeddings model (384 dimensions)."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings
