from pathlib import Path
import json
import os
import faiss
from sentence_transformers import SentenceTransformer
import ollama

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
MODEL_NAME = os.getenv("EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.25"))

class UniversityRAG:
    def __init__(self):
        index_path = ARTIFACTS / "university.faiss"
        chunks_path = ARTIFACTS / "chunks.json"
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError("Index absent. Exécutez : python scripts/build_index.py")
        self.index = faiss.read_index(str(index_path))
        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.embedder = SentenceTransformer(MODEL_NAME, device="cpu")

    def retrieve(self, question: str):
        vector = self.embedder.encode([question], normalize_embeddings=True)
        scores, ids = self.index.search(vector.astype("float32"), TOP_K)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            item = self.chunks[int(idx)].copy()
            item["score"] = float(score)
            results.append(item)
        return results

    def answer(self, question: str):
        docs = self.retrieve(question)
        if not docs or docs[0]["score"] < MIN_SCORE:
            return {"answer": "Je ne trouve pas d'information suffisamment fiable dans les documents disponibles.",
                    "sources": [], "refused": True}
        context = "\n\n".join(
            f"[{d['source']} – chunk {d['chunk_id']}]\n{d['text']}" for d in docs)
        prompt = f"""Tu es un assistant universitaire. Réponds en français.
Utilise exclusivement le CONTEXTE. Si l'information n'y figure pas, dis-le.
N'invente aucune date, règle ou procédure. Cite les identifiants des sources
entre crochets. Les documents peuvent être anciens : signale toute incertitude.

CONTEXTE:
{context}

QUESTION: {question}
RÉPONSE:"""
        response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
        return {"answer": response["response"].strip(),
                "sources": [{"source": d["source"], "chunk_id": d["chunk_id"],
                             "score": d["score"]} for d in docs],
                "refused": False}
