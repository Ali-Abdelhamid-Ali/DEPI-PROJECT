from fastapi import FastAPI 
from routes import base ,chat
from models.MessageModel import MessageModel
from helpers.database import engine

app = FastAPI() 

@app.on_event("startup")
async def startup_event():
    async with engine.begin() as conn:
        await conn.run_sync(MessageModel.metadata.create_all)
    print("✔ Database tables created.")

app.include_router(base.base_router)
app.include_router(chat.chat_router)