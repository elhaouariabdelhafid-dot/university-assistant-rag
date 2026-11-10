
import os
import csv
import json
import time
import statistics
from pathlib import Path

import ollama
from google import genai
from google.genai import types

from src.rag.pipeline import UniversityRAG


# ======================================================
# CONFIGURATION
# ======================================================

MODELS = [
    "qwen2.5:3b",
    "llama3.2:3b",
    "gemini-3.5-flash-lite"
]

RUNS = int(os.getenv("BENCHMARK_RUNS", "1"))

OUTPUT_DIR = Path("data/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUESTIONS = [
    {
        "question": "Quels documents faut-il pour une première inscription ?",
        "expected_source": "DOC02_inscription_reinscription",
        "answerable": True
    },
    {
        "question": "Comment se déroule la réinscription des étudiants ?",
        "expected_source": "DOC02_inscription_reinscription",
        "answerable": True
    },
    {
        "question": "Quel est le salaire du président de l'Université Al Amal ?",
        "expected_source": None,
        "answerable": False
    },
    {
        "question": "Quelle est la couleur du bâtiment principal de l'université ?",
        "expected_source": None,
        "answerable": False
    }
]


# ======================================================
# PROMPT RAG IDENTIQUE POUR LES TROIS MODÈLES
# ======================================================

def build_prompt(question, docs):

    context = "\n\n".join(
        f"[{d['source']}]\n{d['text']}"
        for d in docs
    )

    return f"""
Tu es un assistant universitaire.

Réponds exclusivement en français.

RÈGLES :
1. Utilise uniquement les documents fournis.
2. N'invente aucune information.
3. Réponds précisément à la question.
4. Si une liste est demandée, donne tous les
   éléments présents dans le contexte pertinent.
5. Cite les documents entre crochets.
6. Si l'information est absente, réponds exactement :
   "Information absente des documents."
7. Donne une réponse concise et complète.

DOCUMENTS :
{context}

QUESTION :
{question}

RÉPONSE :
"""


# ======================================================
# QWEN ET LLAMA - OLLAMA
# ======================================================

def generate_ollama(model, prompt):

    start = time.perf_counter()

    response = ollama.generate(
        model=model,
        prompt=prompt,
        options={
            "temperature": 0,
            "num_predict": 240,
            "num_ctx": 4096
        },
        keep_alive="10m"
    )

    latency = time.perf_counter() - start

    tokens = response.eval_count or 0
    generation_time = (
        (response.eval_duration or 0) / 1e9
    )

    speed = (
        tokens / generation_time
        if generation_time > 0 else None
    )

    return {
        "answer": response.response or "",
        "latency": latency,
        "tokens": tokens,
        "tokens_per_second": speed
    }


# ======================================================
# GEMINI - GOOGLE API
# ======================================================

def generate_gemini(client, prompt):

    start = time.perf_counter()

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=240
        )
    )

    latency = time.perf_counter() - start

    usage = response.usage_metadata

    tokens = (
        usage.candidates_token_count
        if usage else None
    )

    return {
        "answer": response.text or "",
        "latency": latency,
        "tokens": tokens,
        "tokens_per_second": None
    }


# ======================================================
# INDICATEURS SIMPLES
# ======================================================

def detect_refusal(answer):

    text = answer.lower()

    expressions = [
        "information absente",
        "ne trouve pas",
        "ne mentionne pas",
        "n'est pas indiqué",
        "n'est pas précisé",
        "aucune information",
        "ne donne aucune information",
        "pas mentionné",
        "pas précisé"
    ]

    return any(
        expression in text
        for expression in expressions
    )


# ======================================================
# BENCHMARK
# ======================================================

def main():

    print("=" * 75)
    print("BENCHMARK RAG : QWEN VS LLAMA VS GEMINI")
    print("=" * 75)

    rag = UniversityRAG()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY n'est pas configurée."
        )

    gemini_client = genai.Client(api_key=api_key)

    results = []

    # Préparer les mêmes passages pour chaque modèle
    prepared = []

    for item in QUESTIONS:

        retrieval_start = time.perf_counter()

        docs = rag.retrieve(item["question"])[:3]

        retrieval_time = (
            time.perf_counter() - retrieval_start
        )

        prepared.append({
            "test": item,
            "prompt": build_prompt(
                item["question"], docs
            ),
            "retrieval_s": retrieval_time,
            "sources": [d["source"] for d in docs]
        })

    for run in range(1, RUNS + 1):

        for item in prepared:

            test = item["test"]

            print("\n" + "=" * 75)
            print("RUN :", run)
            print("QUESTION :", test["question"])
            print("SOURCES FAISS :", item["sources"])

            for model in MODELS:

                print("\n" + "-" * 65)
                print("MODÈLE :", model)

                try:

                    if model.startswith("gemini"):
                        output = generate_gemini(
                            gemini_client,
                            item["prompt"]
                        )
                    else:
                        output = generate_ollama(
                            model,
                            item["prompt"]
                        )

                    answer = output["answer"]
                    refused = detect_refusal(answer)

                    expected_refusal = not test["answerable"]

                    refusal_correct = (
                        refused == expected_refusal
                    )

                    expected_source = test["expected_source"]

                    citation_present = (
                        expected_source in answer
                        if expected_source else None
                    )

                    row = {
                        "run": run,
                        "model": model,
                        "question": test["question"],
                        "answer": answer,
                        "retrieval_s": round(
                            item["retrieval_s"], 3
                        ),
                        "latency_s": round(
                            output["latency"], 3
                        ),
                        "total_rag_s": round(
                            item["retrieval_s"]
                            + output["latency"], 3
                        ),
                        "tokens": output["tokens"],
                        "tokens_per_second": (
                            round(output["tokens_per_second"], 2)
                            if output["tokens_per_second"] is not None
                            else None
                        ),
                        "expected_answerable": test["answerable"],
                        "refusal_detected": refused,
                        "refusal_correct": refusal_correct,
                        "expected_source": expected_source,
                        "citation_present": citation_present,
                        "error": ""
                    }

                    print("\nRÉPONSE :")
                    print(answer)

                    print(
                        f"\nLATENCE : "
                        f"{output['latency']:.2f} s"
                    )

                    print(
                        "TOKENS :", output["tokens"]
                    )

                    print(
                        "TOKENS/S :",
                        output["tokens_per_second"]
                    )

                    print(
                        "REFUS CORRECT (HEURISTIQUE) :",
                        refusal_correct
                    )

                    print(
                        "CITATION ATTENDUE PRÉSENTE :",
                        citation_present
                    )

                except Exception as exc:

                    print("ERREUR :", exc)

                    row = {
                        "run": run,
                        "model": model,
                        "question": test["question"],
                        "answer": "",
                        "retrieval_s": item["retrieval_s"],
                        "latency_s": None,
                        "total_rag_s": None,
                        "tokens": None,
                        "tokens_per_second": None,
                        "expected_answerable": test["answerable"],
                        "refusal_detected": None,
                        "refusal_correct": None,
                        "expected_source": test["expected_source"],
                        "citation_present": None,
                        "error": str(exc)
                    }

                results.append(row)

    # ==================================================
    # SAUVEGARDE JSON
    # ==================================================

    json_path = (
        OUTPUT_DIR / "llm_comparison_results.json"
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ==================================================
    # SAUVEGARDE CSV
    # ==================================================

    csv_path = (
        OUTPUT_DIR / "llm_comparison_results.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    # ==================================================
    # RÉSUMÉ FINAL
    # ==================================================

    print("\n" + "=" * 75)
    print("COMPARAISON FINALE")
    print("=" * 75)

    summary = []

    for model in MODELS:

        rows = [
            r for r in results
            if r["model"] == model and not r["error"]
        ]

        if not rows:
            continue

        latencies = [
            r["latency_s"] for r in rows
        ]

        refusal_score = sum(
            bool(r["refusal_correct"])
            for r in rows
        )

        citations = [
            r["citation_present"]
            for r in rows
            if r["citation_present"] is not None
        ]

        summary.append({
            "model": model,
            "median_latency": statistics.median(
                latencies
            ),
            "refusal_score": refusal_score,
            "total_tests": len(rows),
            "citations": sum(citations),
            "citation_total": len(citations)
        })

    summary.sort(
        key=lambda r: r["median_latency"]
    )

    for rank, item in enumerate(summary, 1):

        print(
            f"{rank}. {item['model']} | "
            f"Latence médiane : "
            f"{item['median_latency']:.2f} s | "
            f"Refus : "
            f"{item['refusal_score']}/{item['total_tests']} | "
            f"Citations : "
            f"{item['citations']}/{item['citation_total']}"
        )

    print("\nFichiers créés :")
    print(json_path)
    print(csv_path)

    print(
        "\nIMPORTANT : les refus et citations sont "
        "évalués par des heuristiques. "
        "La fidélité et la complétude doivent "
        "être vérifiées sur les réponses enregistrées."
    )


if __name__ == "__main__":
    main()
