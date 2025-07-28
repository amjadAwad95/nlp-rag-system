from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from reranker import HuggingFaceCrossEncoderReranker
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from config import PipelineConfig
from .history_pipeline import HistoryPipeline
from langchain_core.messages import HumanMessage, SystemMessage
from database import chat_collection
from bson import ObjectId
from langchain_core.documents import Document
from abc import ABC, abstractmethod

load_dotenv()

class BaseRAGPipeline(ABC):
    def __init__(self):
        self.rag_pipeline = None

    @abstractmethod
    def load_documents(self):
        pass

    @abstractmethod
    def load_chat_history(self):
        pass

    @abstractmethod
    def save_chat_history(self, question, answer):
        pass

    def setup_pipeline(self, config: PipelineConfig = PipelineConfig()):
        if config.chat_type == 'DOC':
            self.documents = self.load_documents()

            text_splitter = SentenceTransformersTokenTextSplitter(
                model_name=config.spliter_model_name,
                tokens_per_chunk=config.tokens_per_chunk,
                chunk_overlap=config.chunk_overlap
            )
            chunks = text_splitter.split_documents(self.documents)

            embedding_model = HuggingFaceEmbeddings(model_name=config.embedding_model_name)

            vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding_model)

        else:
            embedding_model = HuggingFaceEmbeddings(model_name=config.embedding_model_name)
            vectorstore = FAISS.load_local(config.vectorstore_folder_path, embedding_model, allow_dangerous_deserialization=True)

        base_retriever = vectorstore.as_retriever(search_kwargs={"k": config.retriever_k})

        huggingface_reranker = HuggingFaceCrossEncoderReranker(model_name=config.reranker_model_name)
        retriever = huggingface_reranker.get_reranker(base_retriever)

        llm = ChatMistralAI(
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_retries=config.llm_max_retries
        )

        history = HistoryPipeline(config=config.history_config)
        self.rag_pipeline = history.setup(llm, retriever)

    def answer(self, question):
        if not self.rag_pipeline:
            raise RuntimeError("Pipeline not initialized. Call setup_pipeline() first.")

        chat_history = self.load_chat_history()

        response = self.rag_pipeline.invoke({"input": question, "chat_history": chat_history})
        answer = response["answer"]
        context = response["context"]

        self.save_chat_history(question, answer)

        return answer, context

class MemoryRAGPipeline(BaseRAGPipeline):
    """
      A Retrieval-Augmented Generation (RAG) pipeline that retrieves relevant documents,
      reranks them, answers user queries using a language model, and optionally summarizes the response.
      """

    def __init__(self, documents):
        """
        Initialize the RAG pipeline with a list of documents.

        :param documents: A list of LangChain-formatted documents to be used for retrieval.
        :raises ValueError: If no documents are provided.
        """
        super().__init__()
        if not documents:
            raise ValueError("No documents provided, please provide a list of documents, in langchain documents format.")

        self.documents = documents
        self.chat_history = []
        self.summarization_llm = None

    def load_documents(self):
        return self.documents

    def load_chat_history(self):
        return self.chat_history

    def save_chat_history(self, question, answer):
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(SystemMessage(content=answer))

    def summarize(self, raw_answer, summarization_model= "facebook/bart-large-cnn"):
        """
        Summarize a raw answer using a summarization model.

        :param raw_answer: The text to be summarized.
        :param summarization_model: The summarization model to use.
        :return: A summarized version of the input text.
        """
        if not self.summarization_llm:
            hf_pipeline = pipeline("summarization", model=summarization_model)
            self.summarization_llm = HuggingFacePipeline(pipeline=hf_pipeline)

        return self.summarization_llm(raw_answer)


class MongoRAGPipeline(BaseRAGPipeline):
    def __init__(self, chat_id, chat_type):
        super().__init__()

        self.chat_id = chat_id
        self.chat_type = chat_type

    def load_documents(self):
        if self.chat_type == 'NLP':
            self.documents = []
            return self.documents

        self.documents = chat_collection.find_one({"_id": ObjectId(self.chat_id)})["documents"]
        self.documents = [Document(document) for document in self.documents]

        return self.documents

    def load_chat_history(self):
        chat_history_databace = chat_collection.find_one({"_id": ObjectId(self.chat_id)})["chat_history"]

        chat_history = []
        for history in chat_history_databace:
            if history["role"] == "human":
                chat_history.append(HumanMessage(content=history["content"]))
            else:
                chat_history.append(SystemMessage(content=history["content"]))

        return chat_history

    def save_chat_history(self, question, answer):
        chat_collection.update_one(
            {"_id": ObjectId(self.chat_id)},
            {"$push": {"chat_history": {"content": question, "role": "human"}}}
        )

        chat_collection.update_one(
            {"_id": ObjectId(self.chat_id)},
            {"$push": {"chat_history": {"content": answer, "role": "system"}}},
        )