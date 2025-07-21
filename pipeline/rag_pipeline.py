from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from reranker import HuggingFaceCrossEncoderReranker
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from pipeline import PipelineConfig

load_dotenv()


class RAGPipeline:
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
        if not documents:
            raise ValueError("No documents provided, please provide a list of documents, in langchain documents format.")

        self.documents = documents
        self.rag_pipeline = None
        self.summarization_llm = None

    def setup_pipeline(self, config: PipelineConfig=PipelineConfig()):
        """
        Set up the RAG pipeline components including text splitting, embedding generation,
        vector store creation, document reranking, language model initialization, and summarization model.

        :param config: A PipelineConfig instance containing model names, parameters, and options.
        """
        text_splitter = SentenceTransformersTokenTextSplitter(
            tokens_per_chunk=config.tokens_per_chunk,
            chunk_overlap=config.chunk_overlap
        )
        chunks = text_splitter.split_documents(self.documents)

        embedding_model = HuggingFaceEmbeddings(model_name=config.embedding_model_name)

        vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding_model)

        base_retriever = vectorstore.as_retriever(search_kwargs={"k": config.retriever_k})

        huggingface_reranker = HuggingFaceCrossEncoderReranker(model_name=config.reranker_model_name)
        retriever = huggingface_reranker.get_reranker(base_retriever)

        llm = ChatMistralAI(
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_retries=config.llm_max_retries
        )

        self.rag_pipeline = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

        hf_pipeline = pipeline("summarization", model=config.summarization_model)
        self.summarization_llm = HuggingFacePipeline(pipeline=hf_pipeline)

    def answer(self, question):
        """
        Generate an answer to a user question using the RAG pipeline.

        :param question: A user-provided natural language question.
        :return: A tuple containing the generated answer and the list of source documents used.
        :raises RuntimeError: If the pipeline hasn't been initialized with setup_pipeline().
        """
        if not self.rag_pipeline:
            raise RuntimeError("Pipeline not initialized. Call setup_pipeline() first.")

        response = self.rag_pipeline.invoke(question)
        raw_answer = response["result"]
        source_docs = response["source_documents"]

        return raw_answer, source_docs

    def summarize(self, raw_answer):
        """
        Summarize a raw answer using a summarization model.

        :param raw_answer: The text to be summarized.
        :return: A summarized version of the input text.
        :raises RuntimeError: If the summarization model hasn't been initialized.
        """
        if not self.summarization_llm:
            raise RuntimeError("Pipeline not initialized. Call setup_pipeline() first.")

        return self.summarization_llm(raw_answer)

    def __call__(self, question):
        """
        Make the class instance callable. This is a shortcut to the answer() method.

        :param question: A user-provided natural language question.
        :return: A tuple containing the generated answer and the list of source documents used.
        """
        return self.answer(question)
