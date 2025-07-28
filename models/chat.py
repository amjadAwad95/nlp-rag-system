from pydantic import BaseModel, Field
from enum import Enum

class ChatType(str, Enum):
    nlp: str = "NLP"
    doc: str = "DOC"

class Chat(BaseModel):
    chat_type: ChatType = ChatType.nlp
    documents: list[str] = Field(default_factory=list)
    user_id: str = Field(...)
    chat_history: list[str] = Field(default_factory=list)
    language: str = Field("en")

