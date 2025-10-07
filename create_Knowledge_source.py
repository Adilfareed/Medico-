import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Paths
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Load all PDF documents
def load_data(data_path):
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

# Split text into chunks
def create_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return text_splitter.split_documents(docs)

# Create Hugging Face embedding model
def create_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"  # light, fast, good quality
    )

# Create or load FAISS vectorstore
def create_or_load_faiss():
    embedding_model = create_embedding_model()
    documents = load_data(DATA_PATH)
    print(f"📄 Loaded {len(documents)} documents")

    text_chunks = create_chunks(documents)
    print(f"✂️ Created {len(text_chunks)} text chunks")

    try:
        # Try loading existing FAISS index
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        print("✅ Existing FAISS database loaded successfully")
    except Exception as e:
        print("⚠️ Error loading FAISS database:", e)
        print("🧹 Rebuilding FAISS index...")
        # Delete old FAISS folder if it exists
        if os.path.exists(DB_FAISS_PATH):
            shutil.rmtree(DB_FAISS_PATH)

        db = FAISS.from_documents(text_chunks, embedding_model)
        db.save_local(DB_FAISS_PATH)
        print("✅ New FAISS vectorstore created and saved")

    return db

if __name__ == "__main__":
    db = create_or_load_faiss()
    print("🎯 Embedding and FAISS setup complete.")
