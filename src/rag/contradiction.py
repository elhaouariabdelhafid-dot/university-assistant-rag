
import re
import unicodedata


# ============================================================
# NORMALISATION
# ============================================================

def normalize_text(text):

    text = unicodedata.normalize(
        "NFKD",
        text.casefold()
    )

    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# RÈGLE DE CONTRADICTION
# ============================================================

# Cette règle recherche uniquement un délai exprimé
# en heures après la reprise des cours.

DEADLINE_PATTERN = re.compile(
    r"\bdelai\s+"
    r"(?:maximal\s+)?"
    r"de\s+(\d{1,3})\s+heures?\s+"
    r"apres\s+la\s+reprise\s+des\s+cours\b"
)


# ============================================================
# DÉTECTION
# ============================================================

def detect_contradictions(question, documents):

    result = {
        "has_conflict": False,
        "category": None,
        "evidence": [],
        "message": ""
    }

    if not isinstance(question, str):
        return result

    normalized_question = normalize_text(question)

    # Vérifier que la question concerne les absences
    # ou les justificatifs.

    relevant_question = re.search(
        r"\b(absen\w*|justif\w*)\b",
        normalized_question
    )

    if not relevant_question:
        return result

    evidence = []
    seen = set()

    for document in documents:

        source = document.get("source", "")
        chunk_id = document.get("chunk_id", 0)
        text = document.get("text", "")

        normalized_text = normalize_text(text)

        # Éviter les délais sans rapport avec les absences.
        if not re.search(
            r"\b(absen\w*|justif\w*)\b",
            normalized_text
        ):
            continue

        for match in DEADLINE_PATTERN.finditer(
            normalized_text
        ):

            hours = int(match.group(1))

            key = (source, hours)

            if key in seen:
                continue

            seen.add(key)

            evidence.append({
                "source": source,
                "chunk_id": chunk_id,
                "hours": hours,
                "matched_text": match.group(0)
            })

    # Chercher deux valeurs différentes provenant
    # de deux documents distincts.

    conflict_found = False

    for i, first in enumerate(evidence):
        for second in evidence[i + 1:]:

            if (
                first["source"] != second["source"]
                and first["hours"] != second["hours"]
            ):
                conflict_found = True
                break

        if conflict_found:
            break

    if not conflict_found:
        return result

    references = ", ".join(
        f"{item['source']} ({item['hours']} heures)"
        for item in evidence
    )

    result["has_conflict"] = True
    result["category"] = "conflicting_deadlines"
    result["evidence"] = evidence

    result["message"] = (
        "Contradiction documentaire détectée : "
        f"{references}. "
        "Les délais indiqués sont incompatibles. "
        "Une vérification auprès du service "
        "administratif est nécessaire."
    )

    return result
