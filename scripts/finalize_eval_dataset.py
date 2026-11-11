
import json
from collections import Counter
from pathlib import Path

from src.rag.scope_policy import check_scope


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "evaluation"
    / "questions_annotated_draft_v1.jsonl"
)

OUTPUT = (
    ROOT
    / "data"
    / "evaluation"
    / "questions_annotated_v1.jsonl"
)

TEXT_DIR = ROOT / "data" / "processed" / "text"


DOCUMENTARY_IDS = {
    "Q002", "Q003", "Q004", "Q005",
    "Q006", "Q007", "Q008", "Q017"
}

ABSENT_IDS = {
    "Q001",
    "Q009",
    "Q010",
}

POLICY_IDS = {
    "Q011", "Q012", "Q013",
    "Q014", "Q015", "Q016",
}

POLICY_CATEGORIES = {
    "out_of_scope",
    "personal_data",
    "unauthorized_action",
}


# ============================================================
# CHARGEMENT
# ============================================================

def load_dataset():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT}"
        )

    records = []

    with INPUT.open(encoding="utf-8") as file:

        for line_number, line in enumerate(file, 1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON invalide ligne {line_number}"
                ) from error

            records.append(record)

    if len(records) != 17:
        raise ValueError(
            f"17 questions attendues, "
            f"{len(records)} trouvées."
        )

    ids = [record["id"] for record in records]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Identifiants de questions dupliqués."
        )

    expected_ids = (
        DOCUMENTARY_IDS
        | ABSENT_IDS
        | POLICY_IDS
    )

    if set(ids) != expected_ids:
        raise ValueError(
            "La liste des identifiants est incohérente."
        )

    return records


# ============================================================
# VÉRIFICATION DES RÉPONSES DOCUMENTAIRES
# ============================================================

def verify_documentary_answer(record):

    question_id = record["id"]

    if record["category"] != "answerable":
        raise ValueError(
            f"{question_id} : catégorie incorrecte."
        )

    if record["expected_behavior"] != "answer":
        raise ValueError(
            f"{question_id} : comportement incorrect."
        )

    if not record.get("reference_answer"):
        raise ValueError(
            f"{question_id} : réponse absente."
        )

    sources = record.get("expected_sources", [])

    if not sources:
        raise ValueError(
            f"{question_id} : aucune source attendue."
        )

    quotes = record.get("evidence_quotes", [])

    if not quotes:
        raise ValueError(
            f"{question_id} : preuves absentes."
        )

    source_texts = []

    for source in sources:

        path = TEXT_DIR / f"{source}.txt"

        if not path.exists():
            raise FileNotFoundError(path)

        source_texts.append(
            path.read_text(encoding="utf-8")
        )

    for quote in quotes:

        if not isinstance(quote, str) or not quote:
            raise ValueError(
                f"{question_id} : preuve invalide."
            )

        if not any(
            quote in text
            for text in source_texts
        ):
            raise ValueError(
                f"{question_id} : preuve introuvable "
                f"dans les documents : {quote}"
            )

    record["annotation_status"] = "source_checked"

    record["verification_method"] = (
        "exact_evidence_match"
    )

    return record


# ============================================================
# FINALISATION DES QUESTIONS SANS RÉPONSE
# ============================================================

def finalize_absent_question(record):

    question_id = record["id"]

    if record["category"] != "out_of_corpus":
        raise ValueError(
            f"{question_id} : catégorie incorrecte."
        )

    if record["expected_behavior"] != "refuse":
        raise ValueError(
            f"{question_id} : un refus est attendu."
        )

    # Une question documentaire non couverte
    # ne doit pas être bloquée par la politique métier.
    scope = check_scope(record["question"])

    if not scope["allowed"]:
        raise ValueError(
            f"{question_id} : bloquée par la politique "
            "alors qu'un contrôle documentaire est attendu."
        )

    record["expected_sources"] = []

    record["reference_answer"] = (
        "Information absente des documents."
    )

    record["evidence_quote"] = None
    record["evidence_quotes"] = []

    record["annotation_status"] = "reviewed_absent"

    record["verification_method"] = (
        "manual_corpus_review"
    )

    record["annotation_note"] = (
        "Question considérée non documentée "
        "après examen du corpus fictif. "
        "L'absence ne peut pas être prouvée "
        "par une simple recherche de mots-clés."
    )

    return record


# ============================================================
# FINALISATION DES REFUS DE SÉCURITÉ
# ============================================================

def finalize_policy_question(record):

    question_id = record["id"]

    if record["category"] not in POLICY_CATEGORIES:
        raise ValueError(
            f"{question_id} : catégorie incorrecte."
        )

    if record["expected_behavior"] != "refuse":
        raise ValueError(
            f"{question_id} : refus attendu."
        )

    result = check_scope(
        record["question"]
    )

    if result["allowed"]:
        raise ValueError(
            f"{question_id} : question non bloquée."
        )

    if result["category"] != record["category"]:
        raise ValueError(
            f"{question_id} : mauvaise catégorie "
            f"{result['category']}."
        )

    record["expected_sources"] = []

    record["reference_answer"] = result["reason"]

    record["evidence_quote"] = None
    record["evidence_quotes"] = []

    record["annotation_status"] = "policy_checked"

    record["verification_method"] = (
        "scope_policy_test"
    )

    return record


# ============================================================
# VALIDATION FINALE
# ============================================================

def validate_final_dataset(records):

    counts = Counter(
        record["annotation_status"]
        for record in records
    )

    expected_counts = {
        "source_checked": 8,
        "reviewed_absent": 3,
        "policy_checked": 6,
    }

    if dict(counts) != expected_counts:
        raise ValueError(
            f"Répartition incorrecte : {dict(counts)}"
        )

    for record in records:

        if record["expected_behavior"] == "answer":

            if not record["reference_answer"]:
                raise ValueError(
                    f"Réponse manquante : {record['id']}"
                )

            if not record["expected_sources"]:
                raise ValueError(
                    f"Sources manquantes : {record['id']}"
                )

        else:

            if record["expected_sources"]:
                raise ValueError(
                    f"Sources inattendues : {record['id']}"
                )

    return counts


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    records = load_dataset()

    finalized = []

    for record in records:

        record = record.copy()

        question_id = record["id"]

        if question_id in DOCUMENTARY_IDS:

            record = verify_documentary_answer(
                record
            )

        elif question_id in ABSENT_IDS:

            record = finalize_absent_question(
                record
            )

        elif question_id in POLICY_IDS:

            record = finalize_policy_question(
                record
            )

        else:

            raise ValueError(
                f"Question inconnue : {question_id}"
            )

        record["dataset_version"] = "v1"

        finalized.append(record)

    counts = validate_final_dataset(
        finalized
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Écriture dans un fichier temporaire pour
    # éviter une sortie partiellement écrite.
    temp_path = OUTPUT.with_suffix(
        ".jsonl.tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        for record in finalized:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    temp_path.replace(OUTPUT)

    print("=" * 70)
    print("ÉTAPE 7.5 - FINALISATION DU JEU D'ÉVALUATION")
    print("=" * 70)

    print(
        f"Questions totales : {len(finalized)}"
    )

    print(
        "Réponses documentaires :",
        counts["source_checked"]
    )

    print(
        "Informations absentes :",
        counts["reviewed_absent"]
    )

    print(
        "Refus de sécurité :",
        counts["policy_checked"]
    )

    print("\n[OK] Références vérifiées")
    print("[OK] Preuves documentaires présentes")
    print("[OK] Politique de périmètre vérifiée")
    print("[OK] Ancien fichier de brouillon conservé")

    print(f"\nFichier final : {OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
