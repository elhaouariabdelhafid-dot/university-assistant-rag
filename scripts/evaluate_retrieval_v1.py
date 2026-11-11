
import hashlib
import json
import time
from pathlib import Path
from statistics import mean

from src.rag.pipeline import UniversityRAG, MIN_SCORE, TOP_K
from src.rag.scope_policy import check_scope


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    ROOT
    / "data"
    / "evaluation"
    / "questions_annotated_v1.jsonl"
)

OUTPUT = (
    ROOT
    / "data"
    / "evaluation"
    / "retrieval_eval_v1_results.json"
)

POLICY_CATEGORIES = {
    "personal_data",
    "unauthorized_action",
    "out_of_scope",
}


# ============================================================
# CHARGEMENT ET VALIDATION DU JEU
# ============================================================

def load_questions():

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Jeu d'évaluation introuvable : {DATASET}\n"
            "Exécute : python -m scripts.finalize_eval_dataset"
        )

    questions = []

    with DATASET.open(encoding="utf-8") as file:

        for line in file:

            if line.strip():
                questions.append(json.loads(line))

    if len(questions) != 17:
        raise ValueError(
            f"17 questions attendues, "
            f"{len(questions)} trouvées."
        )

    ids = [item["id"] for item in questions]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Identifiants de questions dupliqués."
        )

    return questions


# ============================================================
# CALCUL DU RANG DOCUMENTAIRE
# ============================================================

def first_relevant_rank(documents, expected_sources):

    """
    Retourne le rang du premier chunk provenant
    d'un document attendu.

    La pertinence est mesurée au niveau du document,
    et non au niveau exact du chunk.
    """

    expected = set(expected_sources)

    for rank, document in enumerate(
        documents[:TOP_K],
        start=1
    ):

        if document["source"] in expected:
            return rank

    return None


# ============================================================
# EMPREINTE DU JEU D'ÉVALUATION
# ============================================================

def dataset_sha256():

    return hashlib.sha256(
        DATASET.read_bytes()
    ).hexdigest()


# ============================================================
# ÉVALUATION
# ============================================================

def evaluate():

    if TOP_K != 5:
        raise ValueError(
            "Cette expérience nécessite TOP_K=5. "
            f"Valeur actuelle : {TOP_K}"
        )

    questions = load_questions()

    print("=" * 72)
    print("ÉTAPE 7.6 - ÉVALUATION DU RETRIEVAL RAG")
    print("=" * 72)

    print(f"Questions : {len(questions)}")
    print(f"Top K : {TOP_K}")
    print(f"Seuil FAISS : {MIN_SCORE}")
    print("Aucun appel Gemini.")
    print()

    # Chargement de l'index FAISS existant.
    rag = UniversityRAG()

    results = []

    # Métriques documentaires
    documentary_total = 0
    hit_at_1 = 0
    hit_at_5 = 0
    reciprocal_ranks = []

    # Questions sans réponse documentaire
    absent_total = 0
    absent_below_threshold = 0

    # Politique de périmètre
    policy_total = 0
    policy_correct = 0

    # Temps de retrieval
    retrieval_times_ms = []

    # --------------------------------------------------------
    # PARCOURIR LES QUESTIONS
    # --------------------------------------------------------

    for record in questions:

        question_id = record["id"]
        question = record["question"]
        category = record["category"]

        print("-" * 72)
        print(f"{question_id} : {question}")
        print(f"Catégorie : {category}")

        result = {
            "id": question_id,
            "question": question,
            "category": category,
            "expected_behavior": record["expected_behavior"],
            "expected_sources": record["expected_sources"],
        }

        # ----------------------------------------------------
        # CAS 1 - DEMANDE RELEVANT DU FILTRE DE PÉRIMÈTRE
        # ----------------------------------------------------

        if category in POLICY_CATEGORIES:

            policy_total += 1

            decision = check_scope(question)

            correct = (
                decision["allowed"] is False
                and decision["category"] == category
            )

            if correct:
                policy_correct += 1

            result.update({
                "evaluation_type": "scope_policy",
                "policy_allowed": decision["allowed"],
                "policy_category": decision["category"],
                "policy_correct": correct,
                "retrieval_executed": False,
            })

            print(
                "Politique :",
                "PASS" if correct else "FAIL"
            )

            print("Recherche FAISS : non exécutée")

            results.append(result)
            continue

        # ----------------------------------------------------
        # CAS 2 - RECHERCHE DOCUMENTAIRE
        # ----------------------------------------------------

        start = time.perf_counter()

        documents = rag.retrieve(question)

        retrieval_ms = (
            time.perf_counter() - start
        ) * 1000

        retrieval_times_ms.append(retrieval_ms)

        top_score = (
            float(documents[0]["score"])
            if documents else None
        )

        threshold_accepted = (
            top_score is not None
            and top_score >= MIN_SCORE
        )

        top_documents = [
            {
                "rank": rank,
                "source": document["source"],
                "chunk_id": int(document["chunk_id"]),
                "score": float(document["score"]),
            }
            for rank, document in enumerate(
                documents[:TOP_K],
                start=1
            )
        ]

        result.update({
            "evaluation_type": "retrieval",
            "retrieval_executed": True,
            "retrieval_ms": retrieval_ms,
            "top_score": top_score,
            "threshold_accepted": threshold_accepted,
            "top_documents": top_documents,
        })

        print(
            "Meilleur score :",
            f"{top_score:.4f}"
            if top_score is not None
            else "Aucun"
        )

        # ----------------------------------------------------
        # CAS 2A - QUESTION AVEC RÉPONSE DOCUMENTAIRE
        # ----------------------------------------------------

        if category == "answerable":

            documentary_total += 1

            rank = first_relevant_rank(
                documents,
                record["expected_sources"]
            )

            hit1 = rank == 1
            hit5 = rank is not None

            reciprocal_rank = (
                1.0 / rank
                if rank is not None
                else 0.0
            )

            if hit1:
                hit_at_1 += 1

            if hit5:
                hit_at_5 += 1

            reciprocal_ranks.append(
                reciprocal_rank
            )

            result.update({
                "expected_source_rank": rank,
                "hit_at_1": hit1,
                "hit_at_5": hit5,
                "reciprocal_rank": reciprocal_rank,
            })

            print(
                "Document attendu au rang :",
                rank if rank is not None else "ABSENT"
            )

            print(
                "Hit@5 :",
                "PASS" if hit5 else "FAIL"
            )

        # ----------------------------------------------------
        # CAS 2B - INFORMATION ABSENTE DU CORPUS
        # ----------------------------------------------------

        elif category == "out_of_corpus":

            absent_total += 1

            # Un score inférieur au seuil permet un refus
            # immédiat sans génération.
            #
            # Un score supérieur ne prouve PAS que
            # la réponse existe dans le corpus.

            threshold_refuses = not threshold_accepted

            if threshold_refuses:
                absent_below_threshold += 1

            result.update({
                "threshold_refuses": threshold_refuses,
                "llm_refusal_not_evaluated": True,
            })

            print(
                "Décision du seuil :",
                "REFUSER"
                if threshold_refuses
                else "ACCEPTER LE CONTEXTE"
            )

            if threshold_accepted:
                print(
                    "[ATTENTION] La suffisance documentaire "
                    "reste à vérifier par le pipeline complet."
                )

        else:

            raise ValueError(
                f"Catégorie inconnue : {category}"
            )

        results.append(result)

    # ========================================================
    # CALCUL DES MÉTRIQUES
    # ========================================================

    if documentary_total != 8:
        raise ValueError(
            f"8 questions documentaires attendues, "
            f"{documentary_total} trouvées."
        )

    if absent_total != 3:
        raise ValueError(
            f"3 questions sans réponse attendues, "
            f"{absent_total} trouvées."
        )

    if policy_total != 6:
        raise ValueError(
            f"6 questions de sécurité attendues, "
            f"{policy_total} trouvées."
        )

    metrics = {
        "question_count": len(questions),
        "documentary_question_count": documentary_total,
        "out_of_corpus_question_count": absent_total,
        "policy_question_count": policy_total,

        "hit_at_1": (
            hit_at_1 / documentary_total
        ),

        "document_recall_at_5": (
            hit_at_5 / documentary_total
        ),

        "mrr_at_5": (
            sum(reciprocal_ranks) / documentary_total
        ),

        "scope_policy_accuracy": (
            policy_correct / policy_total
        ),

        "out_of_corpus_threshold_refusal_rate": (
            absent_below_threshold / absent_total
        ),

        "mean_retrieval_ms": (
            mean(retrieval_times_ms)
            if retrieval_times_ms
            else 0.0
        ),
    }

    # ========================================================
    # SAUVEGARDE
    # ========================================================

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "evaluation_name": "university_rag_retrieval_v1",
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": dataset_sha256(),
        "top_k": TOP_K,
        "min_score": MIN_SCORE,
        "metrics": metrics,
        "results": results,
        "limitations": [
            "Document-level metrics, not chunk-level relevance.",
            "Small evaluation set of 17 questions.",
            "No Gemini generation evaluated.",
            "Threshold refusal is not factual correctness.",
            "Scope questions were used during rule development.",
        ],
    }

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8"
    )

    # ========================================================
    # AFFICHAGE
    # ========================================================

    print("\n" + "=" * 72)
    print("RÉSULTATS DE L'ÉVALUATION")
    print("=" * 72)

    print(f"Hit@1 : {metrics['hit_at_1']:.2%}")

    print(
        "Recall@5 documentaire : "
        f"{metrics['document_recall_at_5']:.2%}"
    )

    print(
        f"MRR@5 : {metrics['mrr_at_5']:.4f}"
    )

    print(
        "Politique de périmètre : "
        f"{policy_correct}/{policy_total}"
    )

    print(
        "Questions hors corpus refusées par le seuil : "
        f"{absent_below_threshold}/{absent_total}"
    )

    print(
        "Latence moyenne retrieval : "
        f"{metrics['mean_retrieval_ms']:.2f} ms"
    )

    print("\nIMPORTANT :")
    print("- Aucune réponse générée par Gemini.")
    print("- Les résultats du seuil ne mesurent pas")
    print("  la justesse des refus du LLM.")
    print("- La recherche inclut le calcul des embeddings.")
    print("- L'index FAISS existant n'a pas été modifié.")

    print(f"\nRapport JSON : {OUTPUT}")
    print("=" * 72)


if __name__ == "__main__":
    evaluate()
