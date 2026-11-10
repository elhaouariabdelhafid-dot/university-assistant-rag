from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.rag.pipeline import UniversityRAG

app = FastAPI(title="University Assistant RAG", version="1.0.0")
rag = None

class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000)

@app.on_event("startup")
def startup():
    global rag
    try:
        rag = UniversityRAG()
    except Exception as exc:
        print(f"RAG non initialisé : {exc}")

@app.get("/health")
def health():
    return {"status": "ok", "rag_loaded": rag is not None}

@app.post("/ask")
def ask(payload: Question):
    if rag is None:
        raise HTTPException(status_code=503, detail="Index RAG non disponible.")
    try:
        return rag.answer(payload.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
