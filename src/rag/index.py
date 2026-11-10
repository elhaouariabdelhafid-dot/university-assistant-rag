from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .chunking import load_chunks

ROOT = Path(__file__).resolve().parents[2]
TEXT_DIR = ROOT / "data/processed/text"
ARTIFACTS = ROOT / "artifacts"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def build_index():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    records = load_chunks(TEXT_DIR)
    if not records:
        raise RuntimeError("Aucun chunk. Lance d'abord la collecte et l'extraction.")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    vectors = model.encode([r["text"] for r in records], normalize_embeddings=True,
                           show_progress_bar=True)
    vectors = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(ARTIFACTS / "university.faiss"))
    (ARTIFACTS / "chunks.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Index créé : {index.ntotal} chunks.")

if __name__ == "__main__":
    build_index()
