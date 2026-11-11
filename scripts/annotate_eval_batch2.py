
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    ROOT / "data/evaluation/questions_annotated_draft_v1.jsonl"
)

TEXT_DIR = ROOT / "data/processed/text"


# ============================================================
# ANNOTATIONS VÉRIFIÉES DANS LES DOCUMENTS
# ============================================================

ANNOTATIONS = {
    "Q005": {
        "source": "DOC05_stages_et_PFE",
        "reference_answer": (
            "Le PFE est réalisé en fin de formation. "
            "Il comprend généralement huit étapes : "
            "identifier une problématique, réaliser une étude "
            "bibliographique, analyser les besoins, concevoir "
            "la solution, développer et expérimenter, effectuer "
            "les tests et la validation, rédiger un rapport "
            "et préparer la soutenance."
        ),
        "evidence_quotes": [
            "Identification d'une problématique.",
            "Étude bibliographique.",
            "Analyse des besoins.",
            "Conception de la solution.",
            "Développement et expérimentation.",
            "Tests et validation.",
            "Rédaction d'un rapport.",
            "Préparation d'une soutenance.",
        ],
    },

    "Q006": {
        "source": "DOC06_programmes_formations",
        "reference_answer": (
            "L'Université Al Amal propose des formations "
            "organisées en Licence, Master et Doctorat. "
            "Ses principaux domaines comprennent "
            "l'informatique et le développement logiciel, "
            "l'intelligence artificielle et la science "
            "des données, les réseaux et systèmes "
            "informatiques, les mathématiques et "
            "l'informatique décisionnelle, l'économie "
            "et la gestion, le droit et les sciences "
            "politiques, les langues et la communication, "
            "ainsi que les sciences biologiques et chimiques."
        ),
        "evidence_quotes": [
            (
                "Les formations sont organisées selon "
                "le système Licence, Master et Doctorat."
            ),
            "Informatique et développement logiciel.",
            "Intelligence artificielle et science des données.",
            "Réseaux et systèmes informatiques.",
            "Mathématiques et informatique décisionnelle.",
            "Économie et gestion.",
            "Droit et sciences politiques.",
            "Langues et communication.",
            "Sciences biologiques et chimiques.",
        ],
    },

    "Q007": {
        "source": "DOC07_calendrier_universitaire",
        "reference_answer": (
            "Pour l'année universitaire fictive 2026-2027, "
            "le premier semestre s'étend de septembre "
            "à janvier et le deuxième de février à juin. "
            "Chaque semestre comprend des enseignements, "
            "des contrôles continus et des examens. "
            "Les périodes de vacances peuvent comprendre "
            "les vacances d'hiver, de printemps et les jours "
            "fériés officiels. Les dates précises sont "
            "communiquées par l'administration et peuvent "
            "varier selon les établissements et les filières."
        ),
        "evidence_quotes": [
            (
                "Le premier semestre commence en septembre "
                "et se termine en janvier."
            ),
            (
                "Le deuxième semestre commence en février "
                "et se termine en juin."
            ),
            (
                "Les dates exactes peuvent varier selon "
                "les établissements et les filières."
            ),
            "Contrôles continus.",
            "Examens de fin de semestre.",
            (
                "Les vacances universitaires peuvent comprendre "
                "les vacances d'hiver, les vacances de printemps "
                "et les jours fériés officiels."
            ),
            (
                "Les dates sont communiquées "
                "par l'administration universitaire."
            ),
        ],
    },

    "Q008": {
        "source": "DOC08_frais_procedures",
        "reference_answer": (
            "Un étudiant régulièrement inscrit peut demander "
            "une attestation de scolarité en se connectant "
            "à son espace étudiant, puis en accédant à la "
            "rubrique des documents administratifs. "
            "Il doit remplir le formulaire de demande, "
            "valider sa demande et télécharger l'attestation "
            "lorsqu'elle devient disponible. "
            "Si le service numérique ne fonctionne pas, "
            "il peut contacter le service de scolarité."
        ),
        "evidence_quotes": [
            (
                "Un étudiant régulièrement inscrit peut "
                "demander une attestation de scolarité."
            ),
            "Se connecter à son espace étudiant.",
            "Accéder à la rubrique des documents administratifs.",
            "Remplir le formulaire de demande.",
            "Valider sa demande.",
            "Télécharger l'attestation lorsqu'elle est disponible.",
            (
                "Si le service numérique ne fonctionne pas, "
                "l'étudiant peut contacter le service de scolarité."
            ),
        ],
    },

    "Q017": {
        "source": "DOC01_presentation_universite",
        "reference_answer": (
            "L'Université Al Amal comprend cinq facultés "
            "et écoles : la Faculté des Sciences et Techniques "
            "(FST), la Faculté des Sciences Juridiques, "
            "Économiques et Sociales (FSJES), la Faculté "
            "des Lettres et Sciences Humaines (FLSH), "
            "l'École Nationale des Sciences Appliquées (ENSA) "
            "et la Faculté de Médecine et de Pharmacie (FMP)."
        ),
        "evidence_quotes": [
            "L'université comprend cinq facultés et écoles :",
            "Faculté des Sciences et Techniques (FST).",
            (
                "Faculté des Sciences Juridiques, "
                "Économiques et Sociales (FSJES)."
            ),
            "Faculté des Lettres et Sciences Humaines (FLSH).",
            "École Nationale des Sciences Appliquées (ENSA).",
            "Faculté de Médecine et de Pharmacie (FMP).",
        ],
    },
}


# ============================================================
# VÉRIFICATION DES PREUVES
# ============================================================

def validate_annotation(question_id, annotation):

    source = annotation["source"]
    path = TEXT_DIR / f"{source}.txt"

    if not path.exists():
        raise FileNotFoundError(path)

    document_text = path.read_text(encoding="utf-8")

    for quote in annotation["evidence_quotes"]:

        if quote not in document_text:
            raise ValueError(
                f"{question_id} : preuve introuvable "
                f"dans {source} : {quote}"
            )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    if not DATASET.exists():
        raise FileNotFoundError(
            f"{DATASET}\n"
            "Exécute d'abord : "
            "python -m scripts.annotate_eval_batch1"
        )

    records = []

    with DATASET.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    # Éviter un doublon de Q017 en cas de réexécution.
    records = [
        record for record in records
        if record["id"] != "Q017"
    ]

    if len(records) != 16:
        raise ValueError(
            "Le jeu initial doit contenir 16 questions."
        )

    by_id = {
        record["id"]: record
        for record in records
    }

    if len(by_id) != 16:
        raise ValueError("Identifiants de questions dupliqués.")

    for question_id in ("Q002", "Q003", "Q004"):
        if by_id[question_id]["annotation_status"] != "source_checked":
            raise ValueError(
                f"{question_id} n'est pas encore annotée. "
                "Exécute d'abord le batch 1."
            )

    # --------------------------------------------------------
    # Q001 : mission non explicitement documentée
    # --------------------------------------------------------

    q001 = by_id["Q001"]

    q001["category"] = "out_of_corpus"
    q001["expected_behavior"] = "refuse"
    q001["expected_sources"] = []
    q001["reference_answer"] = (
        "Information absente des documents."
    )
    q001["evidence_quote"] = None
    q001["evidence_quotes"] = []
    q001["annotation_status"] = "reviewed_absent"
    q001["annotation_note"] = (
        "Les huit documents ont été examinés. "
        "Ils ne définissent pas explicitement "
        "la mission institutionnelle de l'université."
    )

    # --------------------------------------------------------
    # Q005 - Q008 : annotations documentaires
    # --------------------------------------------------------

    for question_id in ("Q005", "Q006", "Q007", "Q008"):

        annotation = ANNOTATIONS[question_id]
        validate_annotation(question_id, annotation)

        record = by_id[question_id]

        if annotation["source"] not in record["expected_sources"]:
            raise ValueError(
                f"Source attendue incohérente pour {question_id}"
            )

        record["reference_answer"] = annotation["reference_answer"]
        record["evidence_quote"] = annotation["evidence_quotes"][0]
        record["evidence_quotes"] = annotation["evidence_quotes"]
        record["annotation_status"] = "source_checked"

    # --------------------------------------------------------
    # Q017 : nouvelle question vérifiable sur DOC01
    # --------------------------------------------------------

    annotation = ANNOTATIONS["Q017"]
    validate_annotation("Q017", annotation)

    records.append({
        "id": "Q017",
        "question": (
            "Quelles facultés et écoles composent "
            "l'Université Al Amal ?"
        ),
        "category": "answerable",
        "expected_behavior": "answer",
        "expected_sources": [
            annotation["source"]
        ],
        "reference_answer": annotation["reference_answer"],
        "evidence_quote": annotation["evidence_quotes"][0],
        "evidence_quotes": annotation["evidence_quotes"],
        "annotation_status": "source_checked",
        "annotation_note": (
            "Question ajoutée pour évaluer DOC01 "
            "avec une information explicitement documentée."
        ),
    })

    # --------------------------------------------------------
    # Validation de cohérence avant écriture
    # --------------------------------------------------------

    if len(records) != 17:
        raise AssertionError("17 questions attendues.")

    ids = [record["id"] for record in records]

    if len(ids) != len(set(ids)):
        raise AssertionError("Identifiants dupliqués.")

    for record in records:

        if record["expected_behavior"] == "refuse":
            if record["expected_sources"]:
                raise ValueError(
                    f"Sources inattendues pour {record['id']}"
                )

        if record["annotation_status"] == "source_checked":
            if not record["reference_answer"]:
                raise ValueError(
                    f"Réponse manquante pour {record['id']}"
                )

    # Écriture atomique pour conserver le fichier
    # existant si une erreur survient.
    temp_path = DATASET.with_suffix(".jsonl.tmp")

    with temp_path.open("w", encoding="utf-8") as file:

        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    temp_path.replace(DATASET)

    counts = Counter(
        record["annotation_status"]
        for record in records
    )

    print("=" * 65)
    print("ÉTAPE 7.2 - ANNOTATION Q001 ET Q005 À Q008")
    print("=" * 65)

    print(f"Nombre total de questions : {len(records)}")

    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")

    print("\n[OK] Q005 à Q008 annotées")
    print("[OK] Q017 ajoutée avec preuves DOC01")
    print("[OK] Q001 reclassée : information absente")
    print("[OK] Extraits présents dans les documents sources")
    print("\nAucun appel FAISS ni Gemini.")
    print(f"\nFichier : {DATASET}")
    print("=" * 65)


if __name__ == "__main__":
    main()
