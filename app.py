import streamlit as st
import requests
import json
import time
import tempfile
import os
import pandas as pd
from pypdf import PdfReader
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from typing import Any, List, Mapping, Optional
from tenacity import retry, stop_after_attempt, wait_fixed
from requests.exceptions import RequestException
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure Embedding Configuration
AZURE_EMBEDDING_URL = os.getenv('AZURE_EMBEDDING_URL')
AZURE_EMBEDDING_API_KEY = os.getenv('AZURE_EMBEDDING_API_KEY')
AZURE_EMBEDDING_DEPLOYMENT_NAME = os.getenv('AZURE_EMBEDDING_DEPLOYMENT_NAME')

# LLM Configuration
LLM_BASE_URL = os.getenv('LLM_BASE_URL')
LLM_API_KEY = os.getenv('LLM_API_KEY')

# Custom Azure Embeddings class
class AzureEmbeddings(Embeddings):
    def __init__(self):
        self.url = AZURE_EMBEDDING_URL
        self.api_key = AZURE_EMBEDDING_API_KEY

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = self._get_embedding(text)
            embeddings.append(response)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> List[float]:
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        data = {
            "input": text
        }
        response = requests.post(self.url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
        else:
            raise Exception(f"Error in Azure Embedding API: {response.text}")

# Custom LLM class
class CustomLLM(LLM):
    base_url: str = LLM_BASE_URL
    api_key: str = LLM_API_KEY

    @property
    def _llm_type(self) -> str:
        return "custom"

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "Dify",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except RequestException as e:
            st.error(f"Error communicating with LLM API: {str(e)}")
            return "I'm sorry, but I encountered an error while processing your request. Please try again later."

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"base_url": self.base_url}

# Page configuration
st.set_page_config(page_title="CA Assist", layout="wide", initial_sidebar_state="expanded")

# Custom CSS with Streamlit-compatible theming
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    body {
        font-family: 'Roboto', sans-serif;
        color: var(--text-color);
        background-color: var(--background-color);
    }

    .stApp {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        overflow: hidden;
    }

    .stChatInputContainer {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        width: 80%;
        max-width: 800px;
        background: var(--background-color);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .stChatMessage {
        background: var(--secondary-background-color);
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }

    .stTextInput > div > div > input {
        background: var(--input-background-color);
        color: var(--text-color);
        border: none;
        border-radius: 5px;
    }

    .stButton > button {
        background: var(--secondary-background-color);
        color: var(--text-color);
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: var(--hover-color);
    }

    h1, h2, h3 {
        color: var(--text-color);
    }

    .sidebar .sidebar-content {
        background-color: var(--sidebar-color);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to load and process documents
@st.cache_resource
def process_document(uploaded_file):
    if uploaded_file is None:
        return None

    file_type = uploaded_file.type
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    # Display initial status
    status_placeholder.markdown("📄 Processing document...")
    progress_bar = progress_placeholder.progress(0)

    try:
        # Step 1: Load document
        status_placeholder.markdown("📄 Loading document...")
        progress_bar.progress(20)
        
        if file_type == "application/pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            loader = Docx2txtLoader(tmp_file_path)
            documents = loader.load()
        elif file_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            df = pd.read_excel(uploaded_file)
            documents = [Document(page_content=row.to_string(), metadata={"row": i}) for i, row in df.iterrows()]
        else:
            status_placeholder.error("❌ Unsupported file format")
            progress_placeholder.empty()
            return None

        # Step 2: Split text
        status_placeholder.markdown("✂️ Splitting text into chunks...")
        progress_bar.progress(40)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        # Step 3: Create embeddings
        status_placeholder.markdown("🧠 Creating embeddings...")
        progress_bar.progress(60)
        embeddings = AzureEmbeddings()
        
        # Step 4: Create vector store
        status_placeholder.markdown("📚 Creating vector store...")
        progress_bar.progress(80)
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        if file_type not in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            os.unlink(tmp_file_path)

        # Complete
        progress_bar.progress(100)
        status_placeholder.markdown("✅ Document processed successfully!")
        time.sleep(1)  # Show completion message briefly
        progress_placeholder.empty()
        status_placeholder.empty()
        
        return vectorstore

    except Exception as e:
        status_placeholder.error(f"❌ Error processing document: {str(e)}")
        progress_placeholder.empty()
        return None

# Sidebar
with st.sidebar:
    st.title("Settings")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.rerun()

    # File uploader
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "xlsx", "xls"])
    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            try:
                vectorstore = process_document(uploaded_file)
                st.session_state.vectorstore = vectorstore
                st.success("Document processed successfully!")
            except Exception as e:
                st.error(f"An error occurred while processing the document: {str(e)}")

# Main content
st.title("CA Assist")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Send a message...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    # RAG: Find relevant document chunks
    if hasattr(st.session_state, 'vectorstore') and st.session_state.vectorstore:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        llm = CustomLLM()
        
        with chat_container:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                stop_button_placeholder = st.empty()
                full_response = ""
                stop_gen = stop_button_placeholder.button("Stop Generating")

                try:
                    with st.spinner("Thinking..."):
                        # Get relevant documents
                        docs = retriever.get_relevant_documents(prompt)
                        
                        # Prepare context and messages
                        context = "\n\n".join([doc.page_content for doc in docs])
                        messages = [
                            {"role": "system", "content": f"You are a helpful assistant. Use this context to answer the question: {context}"},
                            {"role": "user", "content": prompt}
                        ]
                        
                        # Make streaming API request
                        headers = {
                            "Authorization": f"Bearer {LLM_API_KEY}",
                            "Content-Type": "application/json"
                        }
                        data = {
                            "model": "Dify",
                            "messages": messages,
                            "stream": True
                        }
                        
                        with requests.post(LLM_BASE_URL, headers=headers, json=data, stream=True) as r:
                            r.raise_for_status()
                            for chunk in r.iter_lines():
                                if stop_gen:
                                    break
                                if chunk:
                                    try:
                                        chunk_data = json.loads(chunk.decode('utf-8').lstrip('data: '))
                                        if chunk_data.get('choices'):
                                            chunk_content = chunk_data['choices'][0]['delta'].get('content', '')
                                            if chunk_content:
                                                full_response += chunk_content
                                                message_placeholder.markdown(full_response + "▌")
                                                time.sleep(0.01)
                                    except json.JSONDecodeError:
                                        pass  # Ignore non-JSON lines

                    message_placeholder.markdown(full_response)
                except Exception as e:
                    st.error(f"An error occurred while processing your request: {str(e)}")
                    full_response = "I'm sorry, but I encountered an error while processing your request. Please try again later."
                    message_placeholder.markdown(full_response)
                finally:
                    stop_button_placeholder.empty()
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        # If no document is uploaded, use the regular chat API
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "Dify",
            "messages": st.session_state.messages,
            "stream": True
        }
        
        with chat_container:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                stop_button_placeholder = st.empty()
                full_response = ""
                stop_gen = stop_button_placeholder.button("Stop Generating")

                try:
                    with st.spinner("Thinking..."):
                        with requests.post(LLM_BASE_URL, headers=headers, json=data, stream=True) as r:
                            r.raise_for_status()
                            for chunk in r.iter_lines():
                                if stop_gen:
                                    break
                                if chunk:
                                    try:
                                        chunk_data = json.loads(chunk.decode('utf-8').lstrip('data: '))
                                        if chunk_data.get('choices'):
                                            chunk_content = chunk_data['choices'][0]['delta'].get('content', '')
                                            if chunk_content:
                                                full_response += chunk_content
                                                message_placeholder.markdown(full_response + "▌")
                                                time.sleep(0.01)
                                    except json.JSONDecodeError:
                                        pass  # Ignore non-JSON lines

                    message_placeholder.markdown(full_response)
                except requests.exceptions.RequestException as e:
                    st.error(f"An error occurred: {e}")
                    full_response = "I'm sorry, but I encountered an error. Please try again later."
                    message_placeholder.markdown(full_response)
                finally:
                    stop_button_placeholder.empty()
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Run the Streamlit app
if __name__ == "__main__":
    st.sidebar.success("CA Assist is ready!")
