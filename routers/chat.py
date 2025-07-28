import os
import tempfile
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from database import chat_collection
from models import Chat
from schema import chat_individual_serial
from utils import load_documents
from bson import ObjectId
from pipeline import MongoRAGPipeline
from typing import Dict
from langdetect import detect
from arabic_config import arabic_pipeline_config

chat_router = APIRouter()

rag_pipeline_cache: Dict[str, MongoRAGPipeline] = {}

@chat_router.get("/chat/{chat_id}")
async def answer_question(chat_id: str, question: str=Query(...)):
    chat = chat_collection.find_one({"_id": ObjectId(chat_id)})

    if not chat:
        raise HTTPException(404, "Chat not found")

    if chat_id not in rag_pipeline_cache:
        rag = MongoRAGPipeline(chat_id, chat["chat_type"])
        if chat["language"] == "ar":
            rag.setup_pipeline(arabic_pipeline_config)
            rag_pipeline_cache[chat_id] = rag

        else:
            rag.setup_pipeline()
            rag_pipeline_cache[chat_id] = rag

    else:
        rag = rag_pipeline_cache[chat_id]

    raw_answer, source_doc = rag.answer(question)

    return {"answer": raw_answer, "source_doc": source_doc}

@chat_router.post('/chats')
async def create_chat(chat: Chat):
    chat_dict = chat.model_dump()

    inserted_chat = chat_collection.insert_one(chat_dict)

    chat_dict['_id'] = inserted_chat.inserted_id

    return chat_individual_serial(chat_dict)


@chat_router.patch('/chats/{chat_id}/')
async def add_document(chat_id: str, files: list[UploadFile] = File(...)):
    is_pdfs = all(file.content_type == 'application/pdf' for file in files)
    if is_pdfs:
        HTTPException(401, detail="Only PDF documents are supported")

    with tempfile.TemporaryDirectory() as temp_dir:
        for file in files:
            temp_file = os.path.join(temp_dir, file.filename)

            with open(temp_file, 'wb') as f:
                f.write(file.file.read())

        documents = load_documents(temp_dir)

    documents = [document.page_content for document in documents]

    text = ""

    for doc in documents:
        text += doc

    main_language = detect(text)

    chat_collection.find_one_and_update({"_id":ObjectId(chat_id)}, {"$set": {"documents": documents, "language": main_language}})

    return [file.filename for file in files]

