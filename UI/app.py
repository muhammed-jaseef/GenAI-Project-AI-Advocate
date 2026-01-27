
import sys
import os

# Adds the project root to path so 'rag' and 'configure' are found
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


import streamlit as st
from rag.chain import get_rag_chain
from rag.web_search import get_web_search_tool
from rag.loader import load_documents
from rag.chunking import chunk_documents
from rag.vector_store import create_vector_store

# Page config
st.set_page_config(page_title="GenAI RAG Chat", layout="wide")
st.title("📄 RAG Chat Application")

# Sidebar settings
st.sidebar.header("Settings")
use_web = st.sidebar.toggle("Enable Web Search", value=False)
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

# Ensure data directory exists
if not os.path.exists("data"):
    os.makedirs("data")

# Chat history initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Process PDF
if uploaded_file:
    pdf_path = os.path.join("data", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Only process if not already in session to avoid re-chunking every click
    if "vector_store_ready" not in st.session_state:
        with st.spinner("Processing PDF..."):
            documents = load_documents(pdf_path)
            chunks = chunk_documents(documents)
            create_vector_store(chunks)
            st.session_state.vector_store_ready = True
        st.sidebar.success("PDF processed and Vector Store ready!")

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
query = st.chat_input("Ask something...")

if query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Thinking..."):
        try:
            # 1. Get the chain with the correct mode
            current_mode = "web" if use_web else "doc"
            chain = get_rag_chain(mode=current_mode)

            # 2. Handle Web Search if enabled
            if use_web:
                web_tool = get_web_search_tool()
                web_results = web_tool.invoke(query) # Using .invoke() for consistency
                
                web_context = "\n".join([r["content"] for r in web_results if "content" in r])
                # We append web results to the query so the LLM sees it as extra context
                full_query = f"Web Search Results:\n{web_context}\n\nUser Question: {query}"
            else:
                full_query = query

            # 3. Invoke chain
            response = chain.invoke(full_query)

            # 4. Display and Save Assistant message
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

        except Exception as e:
            st.error(f"Error: {e}")