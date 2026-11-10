"""Évaluation du système de recherche RAG."""

from pathlib import Path
import json
import time
import statistics

from src.rag.pipeline import UniversityRAG

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/evaluation/questions.json"
OUTPUT = ROOT / "data/evaluation/results.json"

K = 5


def evaluate():
    if not DATASET.exists():
        raise SystemExit("Fichier questions.json introuvable.")

    cases = json.loads(DATASET.read_text(encoding="utf-8"))

    if not cases:
        raise SystemExit("Le jeu de test est vide.")

    rag = UniversityRAG()

    recalls = []
    precisions = []
    reciprocal_ranks = []
    latencies = []
    details = []

    print("\n========== ÉVALUATION DU RETRIEVAL RAG ==========\n")

    for i, case in enumerate(cases, start=1):
        question = case["question"]
        expected = set(case.get("expected_sources", []))

        if not expected:
            print(f"[IGNORÉ] Question {i}: aucune source attendue.")
            continue

        start = time.perf_counter()
        retrieved = rag.retrieve(question)
        latency = (time.perf_counter() - start) * 1000

        retrieved = retrieved[:K]

        # Sources retrouvées parmi les résultats
        found = {
            item["source"]
            for item in retrieved
            if item["source"] in expected
        }

        # Recall : sources attendues retrouvées
        recall = len(found) / len(expected)

        # Precision : chunks provenant d'une source attendue
        relevant_count = sum(
            1 for item in retrieved
            if item["source"] in expected
        )

        precision = relevant_count / len(retrieved) if retrieved else 0.0

        # MRR : rang du premier chunk pertinent
        reciprocal_rank = 0.0

        for rank, item in enumerate(retrieved, start=1):
            if item["source"] in expected:
                reciprocal_rank = 1 / rank
                break

        recalls.append(recall)
        precisions.append(precision)
        reciprocal_ranks.append(reciprocal_rank)
        latencies.append(latency)

        detail = {
            "question": question,
            "expected_sources": sorted(expected),
            "retrieved_sources": [x["source"] for x in retrieved],
            "recall_at_5": recall,
            "precision_at_5": precision,
            "reciprocal_rank_at_5": reciprocal_rank,
            "latency_ms": latency,
        }

        details.append(detail)

        print(f"Question {i}: {question}")
        print(f"Sources attendues : {sorted(expected)}")
        print(f"Sources retrouvées : {[x['source'] for x in retrieved]}")
        print(f"Recall@5 : {recall:.2%}")
        print(f"Precision@5 : {precision:.2%}")
        print(f"MRR@5 : {reciprocal_rank:.3f}")
        print(f"Latence : {latency:.2f} ms")
        print("-" * 55)

    if not recalls:
        raise SystemExit("Aucune question évaluable.")

    # Moyennes globales
    summary = {
        "total_questions": len(recalls),
        "recall_at_5": statistics.mean(recalls),
        "precision_at_5": statistics.mean(precisions),
        "mrr_at_5": statistics.mean(reciprocal_ranks),
        "average_latency_ms": statistics.mean(latencies),
    }

    print("\n========== RÉSULTATS GLOBAUX ==========")
    print(f"Questions évaluées : {summary['total_questions']}")
    print(f"Recall@5 : {summary['recall_at_5']:.2%}")
    print(f"Precision@5 : {summary['precision_at_5']:.2%}")
    print(f"MRR@5 : {summary['mrr_at_5']:.3f}")
    print(f"Latence moyenne : {summary['average_latency_ms']:.2f} ms")

    report = {
        "summary": summary,
        "details": details,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nRapport sauvegardé : {OUTPUT}")


if __name__ == "__main__":
    evaluate()
