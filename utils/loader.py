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


def load_uploaded_documents(uploaded_files):
    """
    Processes PDF files uploaded through Gradio.

    :param uploaded_files (list): A list of Gradio UploadedFile objects.
    :return list: A list of cleaned Document objects.
    """
    documents = []

    for uploaded_file in uploaded_files:
        temp_path = uploaded_file.name
        loader = PyPDFLoader(temp_path)
        loaded_docs = loader.load()

        for doc in loaded_docs:
            doc.page_content = clean(doc.page_content)

        documents.extend(loaded_docs)

    return documents