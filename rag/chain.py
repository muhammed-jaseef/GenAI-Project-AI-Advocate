
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Internal imports based on your exact file structure
from rag.vector_store import load_vector_store
from configure import LLM, LLM_TEMPERATURE, GROQ_API_KEY 

def get_llm():
    """Return the ChatGroq LLM instance."""
    return ChatGroq(
        model=LLM,
        temperature=LLM_TEMPERATURE,
        groq_api_key=GROQ_API_KEY
    )

def format_docs(docs):
    """Formats the retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(mode="doc"):
    """
    Build a modern RAG chain using LCEL:
    - No 'RetrievalQA' to avoid underlines
    - Uses Pipes (|) for clearer logic
    """
    # 1. Setup Vector Store & LLM
    db = load_vector_store()
    if db is None:
        raise ValueError("Vector store not found. Please run vector_store.py first.")

    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()

    # 2. Modern Prompt Template
    template = """
You are a helpful and accurate AI assistant.

Current Mode: {mode}

Rules:
If Mode is "doc":
- Answer ONLY using the provided context.
- If answer is not in context, say "I don't know".

If Mode is "web":
- Use the context and web search results.
- You may use general knowledge.

Context:
{context}

Question:
{question}

Answer:
"""
    # Fill in the 'mode' variable early
    prompt = ChatPromptTemplate.from_template(template).partial(mode=mode)

    # 3. The Modern LCEL Chain (The Pipe | Operator)
    # This replaces RetrievalQA entirely
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# -------------------
# MAIN TEST BLOCK
# -------------------
if __name__ == "__main__":
    print("\n--- 🛠️ Starting Modern rag.chain Test ---")

    test_query = "The base salary"

    try:
        # Create chain (defaults to 'doc' mode)
        chain = get_rag_chain(mode="doc")

        print(f"🤔 Sending Query: {test_query}")
        
        # In LCEL, we use invoke()
        response = chain.invoke(test_query)

        print("\n✅ AI Response:\n")
        print(response)

    except Exception as e:
        print("\n❌ Error while testing rag.chain:")
        print(f"Details: {e}")