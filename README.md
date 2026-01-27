
# 📘 GenAI RAG Chat Application

A Retrieval-Augmented Generation (RAG) based chat application built using LangChain, Groq LLM, FAISS, and Streamlit.  
This app allows users to upload PDFs and ask questions based on document content and optional web search.

---

## 🚀 Features

- 📄 Upload PDF documents
- 🔍 Semantic search using FAISS vector store
- 🤖 AI-powered answers using Groq LLM
- 🌐 Optional web search integration
- 💬 Chat-based interface (Streamlit)
- 📚 Document-based and Web-based modes

---

## 📁 Project Structure

GenAI1/
│
├── __pycache__/
│
├── data/
│
├── faiss_index/
│
├── rag/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── chain.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── loader.py
│   ├── vector_store.py
│   └── web_search.py
│
├── UI/
│   ├── __pycache__/
│   └── app.py              # UI (Streamlit)
│
├── venv/
│
├── .env                   # Environment variables
├── .gitignore
├── configure.py           # Env + settings
├── README.md
└── requirements.txt

## ⚙️ Tech Stack

- Python 3.9+
- LangChain
- Groq API
- FAISS
- HuggingFace Embeddings
- Sentence Transformers
- Streamlit

---

## 📝 How to Use

1. Upload a PDF from the sidebar.
2. Wait for the vector store to be created.
3. Type your question in the input box.
4. Enable Web Search if needed.
5. Get AI-powered answers instantly.


