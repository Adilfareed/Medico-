from langchain_community.document_loaders import PyPDFLoader ,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH="data/"

def load_data(data):
    loader=DirectoryLoader(data,
                             
                    glob='*.pdf',
                    loader_cls=PyPDFLoader
    )
    documentS=loader.load()
    return documentS

documents=load_data(data=DATA_PATH)

print("length",len(documents))


def create_chunks(book_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(book_data)
    return text_chunks

text_cunks=create_chunks(book_data=documents)

print("chunks",len(text_cunks))