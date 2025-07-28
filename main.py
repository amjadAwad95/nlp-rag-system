from fastapi import FastAPI
from routers import user_router, chat_router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.include_router(user_router)
app.include_router(chat_router)