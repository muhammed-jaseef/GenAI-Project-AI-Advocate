
import os
from langchain_community.vectorstores import FAISS
from rag.embeddings import get_embeddings

VECTOR_DB_PATH = "faiss_index"


def create_vector_store(chunks, save_path=VECTOR_DB_PATH):
    """
    Create FAISS vector store from document chunks
    """
    embeddings = get_embeddings()

    db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    db.save_local(save_path)
    return db


def load_vector_store(save_path=VECTOR_DB_PATH):
    """
    Load FAISS vector store from disk
    """
    if not os.path.exists(save_path):
        return None

    embeddings = get_embeddings()
    return FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True
    )



from rag.loader import load_documents
from rag.chunking import chunk_documents  

if __name__ == "__main__":
    #Load & chunk docs
    docs = load_documents("data/EMPLOYEE_AGREEMENT (1).pdf")
    chunks = chunk_documents(docs)

    #Create or load vector store
    db = load_vector_store()
    if db is None:
        db = create_vector_store(chunks)

    #Test search
    results = db.similarity_search("employee salary", k=3)

    print("\nSearch Results:\n")
    for r in results:
        print(r.page_content[:200])
        print("-" * 50)





























