from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

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

# Create HuggingFace embedding model

def create_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embedding_model

embedding_model = create_embedding_model()

# Create and save FAISS vectorstore
DB_FAISS_PATH = "vectorstore/db_faiss"
db =FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)

print("Embedding and FAISS vectorstore creation completed.")
