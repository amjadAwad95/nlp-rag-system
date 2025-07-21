import os
from langchain.document_loaders import PyPDFLoader
from utils import clean

def load_documents(directory="documents"):
    documents_path = os.listdir(directory)
    documents_path = [f"{directory}/{file}" for file in documents_path]

    documents = []
    for file in documents_path:
        loader = PyPDFLoader(file)
        loaded_docs = loader.load()

        for doc in loaded_docs:
            doc.page_content = clean(doc.page_content)

        documents.extend(loaded_docs)

    return documents