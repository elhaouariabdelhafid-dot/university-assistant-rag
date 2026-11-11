
import json
from pathlib import Path

from src.rag.scope_policy import check_scope


ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    ROOT
    / "data"
    / "evaluation"
    / "questions_annotated_draft_v1.jsonl"
)

EXPECTED_POLICY_CATEGORIES = {
    "out_of_scope",
    "personal_data",
    "unauthorized_action",
}


def main():

    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    with DATASET.open(encoding="utf-8") as file:
        questions = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    refusal_questions = [
        record
        for record in questions
        if record["expected_behavior"] == "refuse"
    ]

    total_policy = 0
    passed_policy = 0
    failures = []

    print("=" * 72)
    print("ÉTAPE 7.3 - DIAGNOSTIC DES REFUS")
    print("=" * 72)

    for record in refusal_questions:

        question = record["question"]
        category = record["category"]

        result = check_scope(question)

        print("\n" + "-" * 72)
        print("ID :", record["id"])
        print("QUESTION :", question)
        print("CATÉGORIE ATTENDUE :", category)
        print("POLITIQUE :", result["category"])
        print("AUTORISÉE PAR LE FILTRE :", result["allowed"])

        if category in EXPECTED_POLICY_CATEGORIES:

            total_policy += 1

            success = (
                result["allowed"] is False
                and result["category"] == category
            )

            if success:
                passed_policy += 1
                print("STATUT : PASS")
            else:
                failures.append(record["id"])
                print("STATUT : FAIL")
                print(
                    "[ATTENTION] La politique ne bloque pas "
                    "correctement cette formulation."
                )

        else:

            print(
                "STATUT : À ÉVALUER PAR LE RAG"
            )

            print(
                "Cette question nécessite une vérification "
                "de l'absence de réponse documentaire."
            )

    print("\n" + "=" * 72)
    print("SYNTHÈSE")
    print("=" * 72)

    print(
        "Questions nécessitant un refus :",
        len(refusal_questions)
    )

    print(
        "Questions relevant du filtre de périmètre :",
        total_policy
    )

    print(
        f"Refus corrects au niveau politique : "
        f"{passed_policy}/{total_policy}"
    )

    print(
        "Questions à vérifier par le RAG :",
        len(refusal_questions) - total_policy
    )

    print(
        "Identifiants des échecs :",
        failures if failures else "Aucun"
    )

    print("\nAucun appel FAISS ni Gemini.")
    print(
        "Ce diagnostic ne mesure pas encore "
        "les refus du pipeline complet."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
