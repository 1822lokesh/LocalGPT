import torch
import subprocess
import streamlit as st
from run_localGPT import load_model
from langchain.vectorstores import Chroma
from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer
import os

def model_memory():
    # Adding history to the model.
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
    just say that you don't know, don't try to make up an answer.

    {context}

    {history}
    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    return prompt, memory

# Sidebar contents
with st.sidebar:
    st.title("Converse with your Data")
    add_vertical_space(5)
    st.write("Made with Lokesh")

# Check for available device
if torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
elif torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
else:
    DEVICE_TYPE = "cpu"

# Define document upload functionality
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "csv"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    file_path = os.path.join(PERSIST_DIRECTORY, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded {uploaded_file.name} successfully!")

    # Process the document ingestion in the background
    if "result" not in st.session_state:
        run_langest_commands = ["python", "ingest.py", "--device_type", DEVICE_TYPE]
        result = subprocess.run(run_langest_commands, capture_output=True)
        st.session_state.result = result
        st.success("Document ingestion completed!")

# Initialize embeddings with error handling
if "EMBEDDINGS" not in st.session_state:
    try:
        st.session_state.EMBEDDINGS = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE_TYPE)
    except Exception as e:
        st.error(f"Failed to load embeddings model: {str(e)}")

# Initialize vector store with error handling
if "DB" not in st.session_state and "EMBEDDINGS" in st.session_state:
    try:
        st.session_state.DB = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=lambda x: st.session_state.EMBEDDINGS.encode(x),
            client_settings=CHROMA_SETTINGS,
        )
    except Exception as e:
        st.error(f"Failed to load vector store: {str(e)}")

# Initialize retriever with error handling
if "RETRIEVER" not in st.session_state and "DB" in st.session_state:
    try:
        st.session_state.RETRIEVER = st.session_state.DB.as_retriever()
    except Exception as e:
        st.error(f"Failed to initialize retriever: {str(e)}")

# Load the language model (LLM) with error handling
if "LLM" not in st.session_state:
    try:
        st.session_state["LLM"] = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME)
    except Exception as e:
        st.error(f"Failed to load LLM: {str(e)}")

# Initialize the QA system with error handling
if "QA" not in st.session_state and "LLM" in st.session_state and "RETRIEVER" in st.session_state:
    try:
        prompt, memory = model_memory()
        st.session_state["QA"] = RetrievalQA.from_chain_type(
            llm=st.session_state["LLM"],
            chain_type="stuff",
            retriever=st.session_state.RETRIEVER,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    except Exception as e:
        st.error(f"Failed to initialize QA system: {str(e)}")

# Main app
st.title("LocalGPT App")

# Create a text input box for the user
user_prompt = st.text_input("Input your prompt here")

# If the user hits enter
if user_prompt:
    # Process the user's prompt
    if "QA" in st.session_state:
        try:
            response = st.session_state["QA"](user_prompt)
            answer, docs = response["result"], response["source_documents"]
            st.write(answer)

            # Show the source documents with similarity scores
            with st.expander("Document Similarity Search"):
                if "DB" in st.session_state:
                    search = st.session_state.DB.similarity_search_with_score(user_prompt)
                    for i, doc in enumerate(search):
                        st.write(f"Source Document # {i+1} : {doc[0].metadata['source'].split('/')[-1]}")
                        st.write(doc[0].page_content)
                        st.write("--------------------------------")
        except Exception as e:
            st.error(f"Failed to process the prompt: {str(e)}")
    else:
        st.error("QA system is not initialized.")
