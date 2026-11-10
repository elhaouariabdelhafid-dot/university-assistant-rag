
from pathlib import Path
import json
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

TOP_K = 5


def load_resources():
    index_path = ARTIFACTS / "university.faiss"
    chunks_path = ARTIFACTS / "chunks.json"

    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            "Index ou fichier chunks.json introuvable. "
            "Lance d'abord python -m scripts.build_index"
        )

    index = faiss.read_index(str(index_path))

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    return index, chunks, model


def search(query, index, chunks, model, top_k=TOP_K):
    start = time.perf_counter()

    query_vector = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_vector = np.asarray(
        query_vector,
        dtype="float32"
    )

    k = min(top_k, index.ntotal)

    scores, indices = index.search(query_vector, k)

    elapsed = (time.perf_counter() - start) * 1000

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        results.append({
            "source": chunks[idx]["source"],
            "chunk_id": chunks[idx]["chunk_id"],
            "score": float(score),
            "text": chunks[idx]["text"]
        })

    return results, elapsed


def main():
    index, chunks, model = load_resources()

    print("=" * 65)
    print("TEST DE RECHERCHE SEMANTIQUE - UNIVERSITY ASSISTANT RAG")
    print("=" * 65)

    print(f"Nombre de chunks : {index.ntotal}")
    print(f"Nombre de documents : {len(set(c['source'] for c in chunks))}")

    questions = [
        "Quels documents faut-il pour une première inscription ?",
        "Combien de temps avant l'examen faut-il arriver ?",
        "Quel est le délai pour justifier une absence ?",
        "Quelles sont les étapes du PFE ?",
        "Combien de semestres dure une Licence ?",
        "Quels langages sont étudiés en informatique ?",
        "Quand commence le deuxième semestre ?",
        "Comment obtenir une attestation de scolarité ?"
    ]

    for question in questions:
        results, elapsed = search(
            question, index, chunks, model
        )

        print("\n" + "-" * 65)
        print(f"QUESTION : {question}")
        print(f"Temps de recherche : {elapsed:.2f} ms")

        for rank, result in enumerate(results, start=1):
            print(
                f"\n{rank}. Source : {result['source']}"
                f" | Chunk : {result['chunk_id']}"
                f" | Score : {result['score']:.4f}"
            )

            print(f"   Extrait : {result['text'][:250]}...")


if __name__ == "__main__":
    main()
