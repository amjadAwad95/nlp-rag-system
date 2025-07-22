from dataclasses import dataclass

@dataclass
class HistoryConfig:
    """
    Configuration class for setting up the prompts used in a history-aware RAG pipeline.

    - system_promptPrompt (str): used to rewrite user queries in a self-contained format using chat history.
    - question_answer_system_prompt (str): Prompt used by the assistant to answer a user's question based on retrieved documents.
    """
    system_prompt :str = """
    Given the chat history and the user's latest question,
    rewrite the question to be fully self-contained and understandable without the history.
    Include relevant context from the history if needed. If no rewrite is necessary, return the question as is.
    Do not answer it."""

    question_answer_system_prompt :str = """
    You are an assistant designed to answer questions based on the provided context.
    Use the following retrieved information to generate an accurate and relevant response.
    If the answer cannot be found in the context, respond with 'I don't know.'


    {context}"""