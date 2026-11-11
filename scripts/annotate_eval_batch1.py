
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data/evaluation/questions_candidates_v1.jsonl"
)

OUTPUT = (
    ROOT / "data/evaluation/questions_annotated_draft_v1.jsonl"
)

TEXT_DIR = ROOT / "data/processed/text"


# ============================================================
# RÉPONSES DE RÉFÉRENCE : Q002 À Q004
# ============================================================

ANNOTATIONS = {
    "Q002": {
        "reference_answer": (
            "Pour une première inscription, l'étudiant doit fournir "
            "une copie de sa carte nationale d'identité, "
            "une copie du baccalauréat, le relevé de notes "
            "du baccalauréat, deux photographies d'identité, "
            "un formulaire d'inscription signé et un reçu "
            "de paiement des frais administratifs lorsque "
            "ceux-ci sont applicables. Le dossier doit être "
            "déposé auprès du service de scolarité "
            "de l'établissement concerné."
        ),
        "evidence_quotes": [
            "Copie de la carte nationale d'identité.",
            "Copie du baccalauréat.",
            "Relevé de notes du baccalauréat.",
            "Deux photographies d'identité.",
            "Formulaire d'inscription signé.",
            (
                "Reçu de paiement des frais administratifs, "
                "lorsque ceux-ci sont applicables."
            ),
            (
                "Le dossier doit être déposé auprès du "
                "service de scolarité de l'établissement concerné."
            ),
        ],
    },

    "Q003": {
        "reference_answer": (
            "Les examens sont organisés à la fin de chaque semestre "
            "et peuvent être écrits, oraux, pratiques ou prendre "
            "la forme de projets. L'étudiant doit arriver au moins "
            "15 minutes avant le début de l'épreuve, présenter "
            "sa carte d'étudiant ou une pièce d'identité valide, "
            "respecter la salle et la place attribuées et éteindre "
            "son téléphone. Le matériel autorisé dépend des "
            "consignes de l'enseignant. Les tentatives de fraude "
            "sont signalées et peuvent être transmises à la "
            "commission disciplinaire. Les résultats sont publiés "
            "sur la plateforme après validation pédagogique."
        ),
        "evidence_quotes": [
            "Les examens sont organisés à la fin de chaque semestre.",
            "Les examens peuvent prendre plusieurs formes :",
            "Se présenter au moins 15 minutes avant le début de l'épreuve.",
            "Présenter sa carte d'étudiant ou une pièce d'identité valide.",
            "Respecter la salle et la place qui lui sont attribuées.",
            "Éteindre son téléphone portable avant le début de l'examen.",
            (
                "Le matériel autorisé dépend de la matière "
                "et des consignes de l'enseignant."
            ),
            (
                "Toute tentative de fraude est signalée "
                "au responsable de l'examen."
            ),
            (
                "Les cas peuvent être transmis à la commission "
                "disciplinaire conformément au règlement intérieur."
            ),
            (
                "Les résultats sont publiés sur la plateforme "
                "universitaire après validation par les "
                "responsables pédagogiques."
            ),
        ],
    },

    "Q004": {
        "reference_answer": (
            "L'étudiant doit informer l'administration pédagogique. "
            "Une justification peut nécessiter une demande écrite, "
            "une copie de la carte d'étudiant, un certificat médical "
            "ou un justificatif officiel selon la situation, ainsi "
            "que la date et la nature de l'absence. Le dossier est "
            "transmis au service compétent. Selon ce règlement "
            "fictif, le justificatif doit être déposé dans "
            "un délai maximal de 72 heures après la reprise "
            "des cours."
        ),
        "evidence_quotes": [
            (
                "Un étudiant absent doit informer l'administration "
                "pédagogique selon la procédure de son établissement."
            ),
            "Une demande écrite.",
            "Une copie de la carte d'étudiant.",
            (
                "Un certificat médical ou un justificatif "
                "officiel, selon la situation."
            ),
            "La date et la nature de l'absence.",
            "Le dossier doit être transmis au service compétent.",
            (
                "Dans le cadre de ce règlement fictif, le "
                "justificatif doit être déposé dans un délai "
                "maximal de 72 heures après la reprise des cours."
            ),
        ],
    },
}


# ============================================================
# VÉRIFICATION DES PREUVES
# ============================================================

def validate_evidence(record, quotes):

    sources = record["expected_sources"]

    if not sources:
        raise ValueError(
            f"Aucune source attendue pour {record['id']}"
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
        if not any(
            quote in text
            for text in source_texts
        ):
            raise ValueError(
                f"Preuve introuvable pour {record['id']} : "
                f"{quote}"
            )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    records = []

    with INPUT.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    if len(records) != 16:
        raise ValueError("16 questions attendues.")

    for record in records:

        question_id = record["id"]

        if question_id == "Q001":
            # Ne pas inventer une mission institutionnelle.
            # Il reste quatre documents à examiner.
            record["annotation_status"] = (
                "needs_full_corpus_review"
            )
            record["annotation_note"] = (
                "DOC01 présente l'université mais ne "
                "définit pas explicitement sa mission. "
                "Vérifier DOC05 à DOC08 avant de décider "
                "si la question doit être refusée ou corrigée."
            )
            continue

        if question_id not in ANNOTATIONS:
            continue

        annotation = ANNOTATIONS[question_id]

        quotes = annotation["evidence_quotes"]

        validate_evidence(record, quotes)

        record["reference_answer"] = (
            annotation["reference_answer"]
        )
        record["evidence_quote"] = quotes[0]
        record["evidence_quotes"] = quotes
        record["annotation_status"] = "source_checked"

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                ) + "\n"
            )

    counts = Counter(
        record["annotation_status"]
        for record in records
    )

    print("=" * 65)
    print("ÉTAPE 7.2 - ANNOTATION DES QUESTIONS Q001 À Q004")
    print("=" * 65)

    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")

    print("\n[OK] Extraits vérifiés dans les documents TXT")
    print("[OK] Réponses Q002, Q003 et Q004 enregistrées")
    print("[ATTENTION] Q001 nécessite une vérification complémentaire")
    print(f"\nFichier créé : {OUTPUT}")
    print("=" * 65)


if __name__ == "__main__":
    main()
