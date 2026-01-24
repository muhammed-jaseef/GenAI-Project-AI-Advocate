
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from rag.vector_store import load_vector_store
from configure import LLM, LLM_TEMPERATURE, GROQ_API_KEY


def get_llm():
    return ChatGroq(
        model=LLM,
        temperature=LLM_TEMPERATURE,
        api_key=GROQ_API_KEY
    )


def get_rag_chain():
    # Load FAISS vector store
    db = load_vector_store()
    if db is None:
        raise ValueError("Vector store not found. Create it first.")

    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = get_llm()

    # Custom prompt (Instructor style)
    prompt = ChatPromptTemplate.from_template("""
                                              
You are a helpful and accurate AI assistant.

Mode: {mode}

Rules:

If Mode is "doc":
- Answer ONLY using the provided context.
- If answer is not in context, say "I don't know".
- Do NOT use outside knowledge.

If Mode is "web":
- Use the context and web search results.
- You may use general knowledge.
- Give the most complete and accurate answer.
- Do NOT make up facts.

Context:
{context}

Question:
{question}

Answer:
""")

    # LCEL Chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain

# if __name__ == "__main__":
    # chain = get_rag_chain()
# 
    # question = "termination policy for the employee is what ?"
    # response = chain.invoke(question)
# 
    # print("\nAnswer:\n")
    # print(response.content)












