
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Process PDF
if uploaded_file:
    pdf_path = "data/uploaded.pdf"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.sidebar.success("PDF uploaded successfully!")

    with st.spinner("Processing PDF..."):
        documents = load_documents(pdf_path)
        chunks = chunk_documents(documents)
        create_vector_store(chunks)

    st.sidebar.success("PDF indexed for RAG!")


# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Chat input
query = st.chat_input("Ask something...")


if query:

    # User message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)


    with st.spinner("Thinking..."):

        chain = get_rag_chain()


        # -------- WEB MODE --------
        if use_web:

            web_tool = get_web_search_tool()
            web_results = web_tool.run(query)

            context = ""

            # ✅ FIXED INDENTATION
            for r in web_results:
                if "content" in r and r["content"]:
                    context += r["content"] + "\n"

            # Debug (optional)
            # print("DEBUG context length:", len(context))


        # -------- DOC MODE --------
        else:
            context = ""


        # Invoke chain properly
        final_query = query

        if use_web and context.strip():
            final_query = query + "\n\n" + context

        response = chain.invoke(final_query)

    # Get answer
    answer = response.content if hasattr(response, "content") else str(response)


    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })


    # Display answer
    with st.chat_message("assistant"):
        st.markdown(answer)
