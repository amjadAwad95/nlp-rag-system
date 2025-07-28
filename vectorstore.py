from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from utils import load_documents


documents = load_documents()

text_splitter = SentenceTransformersTokenTextSplitter(
    model_name="sentence-transformers/all-mpnet-base-v2",
    tokens_per_chunk=300,
    chunk_overlap=30
)
chunks = text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding_model)

vectorstore.save_local("rag_system")