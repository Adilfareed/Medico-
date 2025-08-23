import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# Path to PDF data
DATA_PATH = "data/"

# Load all PDF documents from a directory
def load_data(data_path):
    loader = DirectoryLoader(
        data_path,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

documents = load_data(DATA_PATH)
print("Documents loaded:", len(documents))

# Split documents into chunks
def create_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    return chunks

text_chunks = create_chunks(documents)
print("Chunks created:", len(text_chunks))

# Use Google Generative AI Embeddings
def create_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

embedding_model = create_embedding_model()

# Create and save FAISS vectorstore
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)

print("Embedding and FAISS vectorstore creation completed.")
