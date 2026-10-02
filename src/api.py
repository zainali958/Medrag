from fastapi import FastAPI
from pydantic import BaseModel

from rag import RAG

app = FastAPI(title="MedRAG")
rag = RAG()


class Ask(BaseModel):
    question: str


@app.post("/ask")
def ask(body: Ask):
    return rag.answer(body.question)


@app.get("/health")
def health():
    return {"status": "ok"}
