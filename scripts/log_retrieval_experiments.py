
import hashlib
import json
import os
from pathlib import Path

import faiss
import mlflow
import numpy as np
from sentence_transformers import SentenceTransformer

from scripts.compare_retrieval_configs import (
    TEXT_DIR,
    EMBEDDING_MODEL,
    TOP_K,
    CONFIGS,
    QUESTIONS,
    prepare_chunks,
    build_index,
    evaluate,
    measure_faiss_latency,
)


EXPERIMENT_NAME = "university_rag_chunking_comparison"


def corpus_fingerprint(files):
    """Empreinte reproductible du corpus source."""

    digest = hashlib.sha256()

    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())

    return digest.hexdigest()


def questions_fingerprint():
    """Empreinte du jeu de questions d'évaluation."""

    serialized = json.dumps(
        QUESTIONS,
        ensure_ascii=False,
        sort_keys=True,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def main():

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )

    mlflow.set_tracking_uri(tracking_uri)

    faiss.omp_set_num_threads(1)

    files = sorted(TEXT_DIR.glob("*.txt"))

    if not files:
        raise RuntimeError("Corpus TXT introuvable.")

    print("=" * 72)
    print("ÉTAPE 6.3 - SUIVI DES EXPÉRIENCES AVEC MLFLOW")
    print("=" * 72)

    print("Tracking URI :", tracking_uri)
    print("Corpus :", len(files), "documents")
    print("Questions :", len(QUESTIONS))

    mlflow.set_experiment(EXPERIMENT_NAME)

    corpus_hash = corpus_fingerprint(files)
    questions_hash = questions_fingerprint()

    print("\nChargement du modèle d'embeddings...")

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu",
    )

    # Les deux configurations utilisent exactement
    # les mêmes vecteurs de questions.
    question_vectors = model.encode(
        [item["question"] for item in QUESTIONS],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    question_vectors = np.ascontiguousarray(
        question_vectors,
        dtype=np.float32,
    )

    for config_name, config in CONFIGS.items():

        print("\n" + "-" * 72)
        print("CONFIGURATION :", config_name)
        print("-" * 72)

        texts, sources = prepare_chunks(
            files,
            size=config["size"],
            overlap=config["overlap"],
        )

        index, build_time = build_index(
            model,
            texts,
        )

        if index.ntotal != len(sources):
            raise AssertionError(
                "Incohérence index/chunks."
            )

        metrics = evaluate(
            index,
            sources,
            question_vectors,
        )

        latency_ms = measure_faiss_latency(
            index,
            question_vectors,
        )

        with mlflow.start_run(
            run_name=config_name
        ) as run:

            # --------------------------------------------
            # PARAMÈTRES
            # --------------------------------------------

            mlflow.log_params({
                "chunk_size": config["size"],
                "chunk_overlap": config["overlap"],
                "top_k": TOP_K,
                "embedding_model": EMBEDDING_MODEL,
                "index_type": "FAISS_IndexFlatIP",
                "normalized_embeddings": True,
                "document_count": len(files),
                "question_count": len(QUESTIONS),
            })

            # --------------------------------------------
            # MÉTRIQUES
            # --------------------------------------------

            mlflow.log_metrics({
                "chunk_count": len(texts),
                "hit_at_1": metrics["hit_at_1"],
                "recall_at_5": metrics["recall_at_5"],
                "mrr_at_5": metrics["mrr_at_5"],
                "index_build_time_s": build_time,
                "faiss_latency_ms": latency_ms,
            })

            # --------------------------------------------
            # TRAÇABILITÉ
            # --------------------------------------------

            mlflow.set_tags({
                "project": "University Assistant RAG",
                "stage": "6_experimentation",
                "evaluation_level": "document",
                "corpus_type": "synthetic",
                "corpus_sha256": corpus_hash,
                "questions_sha256": questions_hash,
                "llm_used": "false",
            })

            # --------------------------------------------
            # ARTIFACTS
            # --------------------------------------------

            mlflow.log_dict(
                {
                    "configuration": config_name,
                    "parameters": config,
                    "metrics": {
                        "hit_at_1": metrics["hit_at_1"],
                        "recall_at_5": metrics["recall_at_5"],
                        "mrr_at_5": metrics["mrr_at_5"],
                        "faiss_latency_ms": latency_ms,
                    },
                    "questions": metrics["details"],
                },
                "retrieval_evaluation.json",
            )

            print("Run ID :", run.info.run_id)
            print("Chunks :", len(texts))
            print(
                f"Hit@1 : {metrics['hit_at_1']:.2%}"
            )
            print(
                f"Recall@5 : {metrics['recall_at_5']:.2%}"
            )
            print(
                f"MRR@5 : {metrics['mrr_at_5']:.4f}"
            )
            print(
                f"Latence : {latency_ms:.4f} ms"
            )

    print("\n" + "=" * 72)
    print("EXPÉRIENCES MLFLOW ENREGISTRÉES")
    print("=" * 72)
    print(
        "Ouvrir : http://127.0.0.1:5000"
    )
    print(
        "Les index de production n'ont pas été modifiés."
    )


if __name__ == "__main__":
    main()
