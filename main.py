from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.loader import load_markdown_files
from app.splitter import split
from app.vectorstore import build_index
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # index documents once at startup before accepting requests
    docs   = load_markdown_files()
    chunks = split(docs)
    build_index(chunks)
    yield


app = FastAPI(title="RAG-Chatbot", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(router)


@app.get("/")
def root():
    return FileResponse("static/index.html")