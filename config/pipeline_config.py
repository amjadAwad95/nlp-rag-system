from dataclasses import dataclass
from .history_config import HistoryConfig


@dataclass
class PipelineConfig:
    """
    Configuration class for setting up the components of a RAG (Retrieval-Augmented Generation) pipeline.

    - tokens_per_chunk (int): The maximum number of tokens allowed per chunk during text splitting.
    - chunk_overlap (int): The number of tokens to overlap between chunks to preserve context.
    - embedding_model_name (str): The name of the Hugging Face model used for generating embeddings.
    - retriever_k (int): The number of top documents to retrieve from the vector store.
    - reranker_model_name (str): The name of the Hugging Face cross-encoder model used for reranking retrieved documents.
    - llm_model (str): The identifier for the language model used to generate answers.
    - llm_temperature (float): The temperature setting for controlling randomness in LLM output.
    - llm_max_retries (int): Maximum number of retries if the LLM call fails.
    - summarization_model (str): The Hugging Face model used for summarizing the generated answer.
    """
    tokens_per_chunk: int = 300
    chunk_overlap: int = 30
    spliter_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    retriever_k: int = 20
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    llm_model: str = "mistral-medium-latest"
    llm_temperature: float = 0.8
    llm_max_retries: int = 2
    summarization_model: str = "facebook/bart-large-cnn"
    history_config: HistoryConfig = HistoryConfig
