import os
import streamlit as st
import asyncio
import traceback
from dotenv import load_dotenv, find_dotenv

from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment
load_dotenv(find_dotenv())
DB_FAISS_PATH = "vectorstore/db_faiss"


# ✅ Load model asynchronously (without saving)
async def load_model():
    print("Loading model asynchronously...")
    # Load from your local directory
    model = await asyncio.to_thread(SentenceTransformer, "./models/all-MiniLM-L6-v2")
    print("Model loaded successfully!")
    return model

# ✅ Cache the loaded model
@st.cache_resource
def get_model():
    # Streamlit handles async calls in a blocking context
    return asyncio.run(load_model())


# ✅ Custom embedding class
class LocalHuggingFaceEmbeddings(Embeddings):
    def __init__(self):
        try:
            self.model = get_model()
            print("✅ Local embedding model loaded successfully")
        except Exception as e:
            print("❌ Error loading local embedding model:", str(e))
            raise

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()


# ✅ Cache vectorstore to avoid reloading every time
@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = LocalHuggingFaceEmbeddings()
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error("Error loading vectorstore or embeddings:")
        st.text(traceback.format_exc())
        return None


# ✅ Custom prompt
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


# ✅ Streamlit app
def main():
    st.title("🩺 MEDICO Assistant")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question. 
        Also answer questions related to medical and body parts. 
        The answer should be a minimum of 3 lines. 
        If user says hi or something conversational, reply gently.
        If you don't know the answer, just say that you are a medical assistant and please ask relevant questions. 
        If your question is relevant please check spelling. 
        Thank you! Don't try to make up an answer.
        Don't provide anything out of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                    temperature=0.5
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]

            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.text(traceback.format_exc())


if __name__ == "__main__":
    main()
