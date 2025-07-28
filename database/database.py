import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("MONGO_URI")

client = MongoClient(uri)

db = client["rag_system"]

user_collection = db["users"]
chat_collection = db["chats"]

user_collection.create_index("email", unique=True)