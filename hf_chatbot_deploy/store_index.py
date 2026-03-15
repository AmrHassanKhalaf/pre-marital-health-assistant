"""Store index for Rafiqa Pre-Marital Health Assistant knowledge base."""
import os
from dotenv import load_dotenv
from src.helper import (
    load_pdf_file,
    load_files_from_dir,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings,
)
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ─── Configuration ────────────────────────────────
INDEX_NAME = os.environ.get("PINECONE_INDEX", "pre-marital-health-assistant")
NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "pre_marital_health_assistant")

# Resolve paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.environ.get("PDF_PATH", os.path.join(SCRIPT_DIR, "..", "data"))

# ─── Load & Process Data ─────────────────────────
print("📂 Loading data files...")
extracted_data = load_files_from_dir(data_path=PDF_PATH)
print(f"   ✅ Loaded {len(extracted_data)} documents")

print("🔍 Filtering documents...")
filtered_data = filter_to_minimal_docs(extracted_data)

print("✂️  Splitting into chunks...")
text_chunks = text_split(filtered_data, chunk_size=500, chunk_overlap=50)
print(f"   ✅ Created {len(text_chunks)} chunks")

# ─── Initialize Embeddings ───────────────────────
print("🧠 Loading embedding model...")
embeddings = download_hugging_face_embeddings()

# ─── Setup Pinecone ──────────────────────────────
print("🌲 Setting up Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

if not pc.has_index(INDEX_NAME):
    print(f"   Creating new index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"   ✅ Index '{INDEX_NAME}' created")
else:
    print(f"   ℹ️  Index '{INDEX_NAME}' already exists")

index = pc.Index(INDEX_NAME)

# ─── Store Embeddings ────────────────────────────
print("📤 Uploading embeddings to Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=INDEX_NAME,
    embedding=embeddings,
    namespace=NAMESPACE,
)

print(f"✅ Successfully stored {len(text_chunks)} chunks in Pinecone!")
print(f"   Index: {INDEX_NAME}")
print(f"   Namespace: {NAMESPACE}")
print("🎉 Done! Rafiqa knowledge base is ready for pre-marital guidance.")
