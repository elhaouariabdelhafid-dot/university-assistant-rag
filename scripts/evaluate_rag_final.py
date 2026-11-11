
import hashlib
import json
import os
import time
from pathlib import Path
from statistics import mean

import mlflow

from src.rag.pipeline import UniversityRAG


ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "data/evaluation/questions_annotated_v1.jsonl"
OUTPUT = ROOT / "data/evaluation/rag_final_results.json"

EXPERIMENT = "university_rag_final_evaluation"


def load_questions():
    if not DATASET.exists():
        raise FileNotFoundError(
            f"{DATASET} introuvable. "
            "Exécuter python -m scripts.finalize_eval_dataset"
        )

    records = [
        json.loads(line)
        for line in DATASET.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(records) != 17:
        raise ValueError("17 questions attendues.")

    ids = [r["id"] for r in records]

    if len(ids) != len(set(ids)):
        raise ValueError("Identifiants dupliqués.")

    return records


def load_checkpoint():
    if not OUTPUT.exists():
        return {}

    data = json.loads(
        OUTPUT.read_text(encoding="utf-8")
    )

    current_hash = hashlib.sha256(
        DATASET.read_bytes()
    ).hexdigest()

    if data.get("dataset_sha256") != current_hash:
        raise ValueError(
            "Le dataset a changé. Archiver l'ancien "
            "rapport avant une nouvelle évaluation."
        )

    return {
        item["id"]: item
        for item in data.get("results", [])
    }


def save_results(results):
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    payload = {
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": hashlib.sha256(
            DATASET.read_bytes()
        ).hexdigest(),
        "results": list(results.values()),
    }

    temp = OUTPUT.with_suffix(".json.tmp")

    temp.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    temp.replace(OUTPUT)


def source_ids(result):
    sources = result.get("sources") or []

    ids = set()

    for source in sources:
        if isinstance(source, dict):
            value = source.get("source")
        else:
            value = source

        if isinstance(value, str):
            ids.add(value)

    return ids


def score_response(reference, output):
    expected_answer = (
        reference["expected_behavior"] == "answer"
    )

    refused = bool(output.get("refused", False))

    expected_sources = set(
        reference.get("expected_sources", [])
    )

    actual_sources = source_ids(output)

    behavior_correct = (
        not refused if expected_answer else refused
    )

    source_hit = (
        bool(expected_sources & actual_sources)
        if expected_answer
        else None
    )

    citations_present = bool(
        output.get("citations_present", False)
    )

    # Ce contrôle ne mesure pas la justesse sémantique.
    # Elle nécessitera une revue des réponses.
    return {
        "behavior_correct": behavior_correct,
        "source_hit": source_hit,
        "citations_present": citations_present,
        "expected_sources": sorted(expected_sources),
        "actual_sources": sorted(actual_sources),
        "needs_manual_review": expected_answer and not refused,
    }


def compute_metrics(records, results):
    successful = [
        results[r["id"]]
        for r in records
        if (
            r["id"] in results
            and results[r["id"]].get("execution_status")
            == "success"
        )
    ]

    answered = [
        r for r in successful
        if r["category"] == "answerable"
    ]

    refusals = [
        r for r in successful
        if r["expected_behavior"] == "refuse"
    ]

    documentary_sources = [
        r for r in answered
        if r["evaluation"]["source_hit"] is not None
    ]

    latencies = [
        r["total_latency_s"]
        for r in successful
        if isinstance(
            r.get("total_latency_s"),
            (int, float),
        )
    ]

    return {
        "completed_count": len(successful),
        "total_questions": len(records),
        "behavior_accuracy": (
            mean(
                int(r["evaluation"]["behavior_correct"])
                for r in successful
            )
            if successful else 0.0
        ),
        "refusal_accuracy": (
            mean(
                int(r["evaluation"]["behavior_correct"])
                for r in refusals
            )
            if refusals else 0.0
        ),
        "documentary_behavior_accuracy": (
            mean(
                int(r["evaluation"]["behavior_correct"])
                for r in answered
            )
            if answered else 0.0
        ),
        "source_hit_rate": (
            mean(
                int(r["evaluation"]["source_hit"])
                for r in documentary_sources
            )
            if documentary_sources else 0.0
        ),
        "mean_total_latency_s": (
            mean(latencies)
            if latencies else 0.0
        ),
    }


def main():
    records = load_questions()
    results = load_checkpoint()

    print("=" * 72)
    print("ÉTAPE 7.7 - ÉVALUATION RAG COMPLÈTE")
    print("=" * 72)

    print("Questions :", len(records))
    print("Déjà exécutées :", len(results))
    print("Fournisseur :", os.getenv("LLM_PROVIDER", "gemini"))

    rag = UniversityRAG()

    for index, record in enumerate(records, 1):
        qid = record["id"]

        if (
            qid in results
            and results[qid].get("execution_status")
            == "success"
        ):
            print(f"[SKIP] {qid} déjà évaluée")
            continue

        print("\n" + "-" * 72)
        print(f"{index}/17 - {qid}")
        print("QUESTION :", record["question"])

        try:
            start = time.perf_counter()

            output = rag.answer(
                record["question"]
            )

            elapsed = time.perf_counter() - start

            evaluation = score_response(
                record,
                output
            )

            results[qid] = {
                "id": qid,
                "question": record["question"],
                "category": record["category"],
                "expected_behavior": record[
                    "expected_behavior"
                ],
                "execution_status": "success",
                "answer": output.get("answer", ""),
                "refused": bool(
                    output.get("refused", False)
                ),
                "model": output.get("model"),
                "total_latency_s": elapsed,
                "evaluation": evaluation,
            }

            print("REFUSED :", results[qid]["refused"])
            print(
                "COMPORTEMENT :",
                "PASS"
                if evaluation["behavior_correct"]
                else "FAIL"
            )
            print("SOURCES :", evaluation["actual_sources"])
            print("RÉPONSE :", output.get("answer", ""))

        except Exception as error:
            results[qid] = {
                "id": qid,
                "question": record["question"],
                "category": record["category"],
                "expected_behavior": record[
                    "expected_behavior"
                ],
                "execution_status": "error",
                "error_type": type(error).__name__,
                "error_message": str(error)[:300],
            }

            save_results(results)

            print(
                "[ERREUR]",
                type(error).__name__,
                str(error)[:200],
            )
            print(
                "Résultats conservés. "
                "Relancer ce script pour reprendre."
            )
            raise

        save_results(results)

    metrics = compute_metrics(records, results)

    print("\n" + "=" * 72)
    print("RÉSULTATS FINAUX")
    print("=" * 72)

    for name, value in metrics.items():
        print(f"{name}: {value}")

    if metrics["completed_count"] != len(records):
        raise RuntimeError(
            "Évaluation incomplète : "
            "ne pas publier des métriques finales."
        )

    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5000",
        )
    )

    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(
        run_name="rag_17_questions_v1"
    ) as run:

        mlflow.log_params({
            "question_count": 17,
            "dataset_version": "v1",
            "retriever": "FAISS_IndexFlatIP",
            "top_k": 5,
            "embedding_model": (
                "sentence-transformers/"
                "paraphrase-multilingual-MiniLM-L12-v2"
            ),
            "llm_provider": os.getenv(
                "LLM_PROVIDER",
                "gemini",
            ),
        })

        mlflow.log_metrics({
            k: float(v)
            for k, v in metrics.items()
        })

        mlflow.log_dict(
            {
                "metrics": metrics,
                "results": list(results.values()),
            },
            "rag_evaluation.json",
        )

        print("MLflow Run ID :", run.info.run_id)

    print("\n[OK] Évaluation terminée.")
    print("[OK] Résultats sauvegardés :", OUTPUT)
    print("[IMPORTANT] Revoir manuellement les réponses")
    print("documentaires avant toute conclusion factuelle.")


if __name__ == "__main__":
    main()
