from langchain_community.document_loaders import PyPDFLoader

def load_documents(path: str):
    loader = PyPDFLoader(path)
    return loader.load()


if __name__ == "__main__":
   docs = load_documents("data/EMPLOYEE_AGREEMENT (1).pdf")
   print(docs)
   print("Number of pages:", len(docs))
