from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import HistoryConfig


class HistoryPipeline:
    """
     A history-aware RAG pipeline that enhances user queries using chat history,
     retrieves relevant documents based on the rewritten query, and generates a grounded answer.

     This class combines query contextualization (using chat history), retrieval from a vector store,
     and document-based question answering into a single callable pipeline. It is designed to be integrated
     into a broader RAG application where user interaction occurs over multiple turns.

     Typical usage includes:
       - Using chat history to rephrase follow-up questions.
       - Retrieving documents relevant to the rewritten question.
       - Answering the user's question based on retrieved documents and chat context.
     """
    def __init__(self, config: HistoryConfig = HistoryConfig()):
        """
        Initialize the HistoryPipeline with a configuration object.

        :param config: An instance of HistoryConfig containing the system and QA prompts.
        """
        self.config = config


    def setup(self, llm, retriever):
        """
        Set up the components of the history-aware RAG pipeline:
        - A retriever that reformulates questions using chat history.
        - A chain that answers the reformulated question using retrieved context.
        - A combined retrieval+answering pipeline that handles both steps end-to-end.

        :param llm: A language model (LLM) that supports message-based interaction (e.g. ChatOpenAI, ChatMistralAI).
        :param retriever: A document retriever (e.g. FAISS retriever or a reranked retriever).
        :return: A RAG pipeline (Runnable) that can be invoked with {"input": question, "chat_history": history}.
        """
        contextualize_question_prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        self.history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_question_prompt)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self.config.question_answer_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        self.question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        self.rag_pipeline = create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)

        return self.rag_pipeline
