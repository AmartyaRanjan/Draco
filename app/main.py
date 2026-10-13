from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.query import router as query_router
from app.api.routes import router
from app.db.registry import init_db

app = FastAPI(
    title="RAG Ingestion API",
    description="Upload documents for asynchronous RAG ingestion.",
    version="0.1.0",
)

app.include_router(router)
app.include_router(query_router)
app.include_router(chat_router)


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}