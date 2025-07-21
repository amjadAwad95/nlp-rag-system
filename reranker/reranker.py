from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

class HuggingFaceCrossEncoderReranker:
    """
    Reranks documents using a Hugging Face cross-encoder model.
    """
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        """
        Initialize the cross-encoder reranker with a specified Hugging Face model.
        :param model_name: Name of the Hugging Face cross-encoder model to use.
        """
        self.model_name = model_name
        self.model = HuggingFaceCrossEncoder(model_name=model_name)
        self.compression_retriever_cross = None

    def get_reranker(self, retriever, top_n=4):
        """
        Create or return a ContextualCompressionRetriever using cross-encoder reranking.
        :param retriever: The base retriever
        :param top_n: Number of top documents to keep after reranking.
        :return: A ContextualCompressionRetriever with cross-encoder reranking.
        """
        if self.compression_retriever_cross and self.compression_retriever_cross.base_compressor.top_n == top_n:
            return self.compression_retriever_cross

        compressor_cross = CrossEncoderReranker(model=self.model, top_n=top_n)
        self.compression_retriever_cross = ContextualCompressionRetriever(
            base_compressor=compressor_cross, base_retriever=retriever
        )

        return self.compression_retriever_cross

