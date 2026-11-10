
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    ROOT / "data"
    / "evaluation"
    / "questions_candidates_v1.jsonl"
)

TEXT_DIR = ROOT / "data" / "processed" / "text"

VALID_CATEGORIES = {
    "answerable",
    "out_of_corpus",
    "out_of_scope",
    "personal_data",
    "unauthorized_action",
}

REQUIRED_FIELDS = {
    "id",
    "question",
    "category",
    "expected_behavior",
    "expected_sources",
    "reference_answer",
    "evidence_quote",
    "annotation_status",
}


def main():

    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    valid_sources = {
        path.stem
        for path in TEXT_DIR.glob("*.txt")
    }

    records = []

    with DATASET.open(encoding="utf-8") as file:

        for line_number, line in enumerate(file, 1):

            if not line.strip():
                continue

            record = json.loads(line)

            missing = REQUIRED_FIELDS - set(record)

            if missing:
                raise ValueError(
                    f"Ligne {line_number}: "
                    f"champs absents {missing}"
                )

            records.append(record)

    if len(records) != 16:
        raise ValueError(
            f"16 questions attendues, {len(records)} trouvées."
        )

    ids = [r["id"] for r in records]
    questions = [r["question"].strip() for r in records]

    if len(ids) != len(set(ids)):
        raise ValueError("Identifiants dupliqués.")

    if len(questions) != len(set(questions)):
        raise ValueError("Questions dupliquées.")

    for record in records:

        category = record["category"]
        behavior = record["expected_behavior"]
        sources = record["expected_sources"]

        if not isinstance(record["question"], str):
            raise ValueError("Question invalide.")

        if not record["question"].strip():
            raise ValueError("Question vide.")

        if category not in VALID_CATEGORIES:
            raise ValueError(f"Catégorie inconnue : {category}")

        if behavior not in {"answer", "refuse"}:
            raise ValueError(f"Comportement inconnu : {behavior}")

        if not isinstance(sources, list):
            raise ValueError("expected_sources doit être une liste.")

        if category == "answerable":

            if behavior != "answer" or not sources:
                raise ValueError(
                    f"Annotation incohérente : {record['id']}"
                )

        else:

            if behavior != "refuse" or sources:
                raise ValueError(
                    f"Annotation incohérente : {record['id']}"
                )

        for source in sources:

            if source not in valid_sources:
                raise ValueError(
                    f"Source absente du corpus : {source}"
                )

        if record["annotation_status"] != "pending_review":
            raise ValueError(
                "Ce validateur attend des annotations candidates."
            )

    counts = Counter(
        r["category"]
        for r in records
    )

    print("=" * 65)
    print("VALIDATION DU JEU D'ÉVALUATION")
    print("=" * 65)

    print(f"Questions : {len(records)}")
    print(f"Documents du corpus : {len(valid_sources)}")

    for category, count in sorted(counts.items()):
        print(f"{category}: {count}")

    print("\n[OK] Format JSONL valide")
    print("[OK] Identifiants uniques")
    print("[OK] Questions uniques")
    print("[OK] Catégories et comportements cohérents")
    print("[OK] Identifiants de sources existants")

    print("\nATTENTION :")
    print("La présence réelle des réponses dans les documents")
    print("et leur exactitude restent à vérifier manuellement.")

    print("\nVALIDATION STRUCTURELLE RÉUSSIE")


if __name__ == "__main__":
    main()
