
import json
import os
import statistics
import time
from pathlib import Path

import mlflow
from google.genai.errors import ServerError

from src.rag.pipeline import UniversityRAG


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "evaluation"

QUESTIONS_PATH = EVAL_DIR / "questions.json"

PROGRESS_PATH = (
    EVAL_DIR / "generation_improved_progress.json"
)

RESULTS_PATH = (
    EVAL_DIR / "generation_improved_results.json"
)

MODEL = os.getenv(
    "GEMINI_MODEL", "gemini-3.5-flash-lite"
)

CONFIG = {
    "model": MODEL,
    "provider": "gemini",
    "pipeline_version": "context-expansion-v1",
    "top_k": int(os.getenv("TOP_K", "5")),
    "generation_top_k": int(
        os.getenv("GENERATION_TOP_K", "3")
    ),
    "min_relevance_score": float(
        os.getenv("MIN_RELEVANCE_SCORE", "0.25")
    ),
}

# =====================================================
# QUESTIONS D'ÉVALUATION
# =====================================================

questions = json.loads(
    QUESTIONS_PATH.read_text(encoding="utf-8")
)

TESTS = [
    {
        "question": item["question"],
        "expected_refused": False,
        "expected_sources": item["expected_sources"]
    }
    for item in questions
]

TESTS.extend([
    {
        "question": (
            "Quel est le salaire du président "
            "de l'Université Al Amal ?"
        ),
        "expected_refused": True,
        "expected_sources": []
    },
    {
        "question": (
            "Quelle est la couleur du bâtiment "
            "principal de l'université ?"
        ),
        "expected_refused": True,
        "expected_sources": []
    }
])


# =====================================================
# SAUVEGARDE ET REPRISE
# =====================================================

def save_progress(results):
    payload = {
        "config": CONFIG,
        "results": results
    }

    temporary_path = PROGRESS_PATH.with_suffix(".tmp")

    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    temporary_path.replace(PROGRESS_PATH)


def load_progress():
    if not PROGRESS_PATH.exists():
        return []

    payload = json.loads(
        PROGRESS_PATH.read_text(encoding="utf-8")
    )

    if payload.get("config") != CONFIG:
        raise RuntimeError(
            "La configuration a changé. "
            "Ne réutilise pas l'ancien checkpoint "
            "pour une expérience différente."
        )

    return payload["results"]


# =====================================================
# CALCUL DES MÉTRIQUES
# =====================================================

def calculate_metrics(results):

    total = len(results)

    documented = [
        r for r in results
        if not r["expected_refused"]
    ]

    latencies = [r["latency_s"] for r in results]

    return {
        "test_success_rate": (
            sum(r["success"] for r in results) / total
        ),
        "refusal_accuracy": (
            sum(r["refusal_correct"] for r in results)
            / total
        ),
        "source_hit_rate": (
            sum(r["source_correct"] for r in documented)
            / len(documented)
            if documented else 0.0
        ),
        "mean_latency_s": statistics.mean(latencies),
        "median_latency_s": statistics.median(latencies),
        "min_latency_s": min(latencies),
        "max_latency_s": max(latencies),
        "num_questions": total
    }


# =====================================================
# ÉVALUATION PRINCIPALE
# =====================================================

def main():

    print("=" * 70)
    print("ÉVALUATION RAG - GEMINI + EXPANSION CONTEXTUELLE")
    print("=" * 70)

    if os.getenv("LLM_PROVIDER", "gemini") != "gemini":
        raise RuntimeError(
            "Cette expérience doit utiliser Gemini."
        )

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY n'est pas configurée."
        )

    results = load_progress()

    completed = {
        r["test_number"] for r in results
    }

    print(
        f"Progression retrouvée : "
        f"{len(completed)}/{len(TESTS)}"
    )

    rag = UniversityRAG()

    for index, test in enumerate(TESTS, 1):

        if index in completed:
            print(f"\nTEST {index} : déjà enregistré")
            continue

        print("\n" + "=" * 70)
        print(f"TEST {index}/{len(TESTS)}")
        print("QUESTION :", test["question"])

        start = time.perf_counter()

        try:
            result = rag.answer(test["question"])

        except ServerError as exc:

            if getattr(exc, "code", None) != 503:
                raise

            print("\n[API GEMINI] ERREUR 503")
            print("Le service est temporairement indisponible.")
            print(
                "Résultats sauvegardés : "
                f"{len(results)}/{len(TESTS)}"
            )
            print(
                "Relance la même commande pour reprendre."
            )

            save_progress(results)
            return

        elapsed = time.perf_counter() - start

        actual_sources = [
            source["source"]
            for source in result["sources"]
        ]

        refusal_correct = (
            result["refused"]
            == test["expected_refused"]
        )

        if test["expected_refused"]:
            source_correct = len(actual_sources) == 0
        else:
            source_correct = bool(
                set(test["expected_sources"])
                & set(actual_sources)
            )

        success = refusal_correct and source_correct

        row = {
            "test_number": index,
            "question": test["question"],
            "answer": result["answer"],
            "expected_refused": test["expected_refused"],
            "actual_refused": result["refused"],
            "refusal_correct": refusal_correct,
            "expected_sources": test["expected_sources"],
            "actual_sources": actual_sources,
            "source_correct": source_correct,
            "success": success,
            "latency_s": elapsed,
            "retrieval_s": result["retrieval_s"],
            "generation_s": result["latency_s"],
            "model": result["model"]
        }

        results.append(row)

        # Sauvegarde immédiate après chaque question
        save_progress(results)

        print("\nRÉPONSE :", result["answer"])
        print("REFUS CORRECT :", refusal_correct)
        print("SOURCE CORRECTE :", source_correct)
        print(f"TEMPS : {elapsed:.3f} s")
        print("STATUT :", "PASS" if success else "FAIL")

        print(
            f"Progression sauvegardée : "
            f"{len(results)}/{len(TESTS)}"
        )

    # =================================================
    # RAPPORT FINAL
    # =================================================

    results.sort(key=lambda r: r["test_number"])

    if len(results) != len(TESTS):
        print("Évaluation incomplète.")
        return

    metrics = calculate_metrics(results)

    report = {
        "model": MODEL,
        "config": CONFIG,
        "metrics": metrics,
        "results": results
    }

    RESULTS_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # =================================================
    # MLFLOW
    # =================================================

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000"
    )

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(
        "university-rag-generation"
    )

    with mlflow.start_run(
        run_name="gemini-generation-improved-context"
    ) as run:

        mlflow.log_params(CONFIG)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(RESULTS_PATH))

        print("\nRUN ID :", run.info.run_id)

    # =================================================
    # AFFICHAGE FINAL
    # =================================================

    print("\n" + "=" * 70)
    print("ÉVALUATION TERMINÉE")
    print("=" * 70)

    for name, value in metrics.items():
        print(f"{name} : {value:.4f}")

    print("\nRapport :", RESULTS_PATH)
    print("Expérience enregistrée dans MLflow.")


if __name__ == "__main__":
    main()
