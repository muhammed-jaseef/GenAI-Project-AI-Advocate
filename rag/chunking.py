
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    from rag.loader import load_documents
    docs = load_documents("data/EMPLOYEE_AGREEMENT (1).pdf")
    chunks = chunk_documents(docs)

    print("Total chunks:", len(chunks))
    print("\nFirst chunk preview:\n")
    print(chunks[0].page_content)

