
from pathlib import Path
import csv
import hashlib
import re


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

TEXT_DIR = ROOT / "data" / "processed" / "text"
METADATA_DIR = ROOT / "data" / "metadata"

OUTPUT = METADATA_DIR / "documents.csv"

METADATA_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# FONCTIONS
# =====================================================

def calculate_sha256(path):
    """Calculer l'empreinte SHA-256 d'un document."""

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)

    return sha256.hexdigest()


def count_words(text):
    """Compter les mots du document."""

    return len(text.split())


def check_quality(text):
    """Effectuer des contrôles simples de qualité."""

    issues = []

    if not text.strip():
        issues.append("empty_document")

    if len(text.strip()) < 100:
        issues.append("very_short_document")

    if "\x00" in text:
        issues.append("null_character")

    if "\ufffd" in text:
        issues.append("replacement_character")

    if issues:
        return "warning", ";".join(issues)

    return "ok", ""


# =====================================================
# CONSTRUCTION DE L'INVENTAIRE
# =====================================================

def main():

    print("=" * 70)
    print("INVENTAIRE DOCUMENTAIRE - UNIVERSITY RAG")
    print("=" * 70)

    files = sorted(TEXT_DIR.glob("DOC*.txt"))

    print(f"\nDocuments trouvés : {len(files)}")

    if not files:
        raise RuntimeError(
            "Aucun document trouvé dans data/processed/text/"
        )

    documents = []
    observed_hashes = {}
    observed_ids = set()

    for path in files:

        document_id = path.stem.split("_")[0]

        if not re.fullmatch(r"DOC\d+", document_id):
            raise ValueError(
                f"Identifiant invalide : {path.name}"
            )

        if document_id in observed_ids:
            raise ValueError(
                f"Identifiant dupliqué : {document_id}"
            )

        observed_ids.add(document_id)

        raw_bytes = path.read_bytes()

        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Encodage UTF-8 invalide : {path.name}"
            ) from exc

        sha256 = calculate_sha256(path)
        quality_status, quality_issues = check_quality(text)

        if sha256 in observed_hashes:
            quality_status = "warning"
            duplicate = observed_hashes[sha256]
            quality_issues = ";".join(
                filter(
                    None,
                    [quality_issues, f"duplicate_of:{duplicate}"]
                )
            )
        else:
            observed_hashes[sha256] = document_id

        document = {
            "document_id": document_id,
            "filename": path.name,
            "relative_path": path.relative_to(ROOT).as_posix(),
            "format": "txt",
            "source_type": "synthetic_test_document",
            "origin": "Corpus fictif Universite Al Amal",
            "source_url": "",
            "publication_date": "",
            "version": "unknown",
            "temporal_status": "unknown",
            "sha256": sha256,
            "size_bytes": len(raw_bytes),
            "character_count": len(text),
            "word_count": count_words(text),
            "quality_status": quality_status,
            "quality_issues": quality_issues,
        }

        documents.append(document)

        print("\n------------------------------------------")
        print("Document :", document_id)
        print("Fichier :", path.name)
        print("Taille :", len(raw_bytes), "octets")
        print("Mots :", document["word_count"])
        print("SHA-256 :", sha256[:20] + "...")
        print("Qualité :", quality_status)

        if quality_issues:
            print("Alertes :", quality_issues)

    # =================================================
    # EXPORT CSV
    # =================================================

    with open(
        OUTPUT,
        "w",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        writer = csv.DictWriter(
    csvfile,
    fieldnames=list(documents[0].keys()),
    lineterminator="\n"
)

        writer.writeheader()
        writer.writerows(documents)

    # =================================================
    # RÉSUMÉ
    # =================================================

    print("\n" + "=" * 70)
    print("INVENTAIRE TERMINÉ")
    print("=" * 70)

    print("Nombre de documents :", len(documents))

    print(
        "Documents sans alerte :",
        sum(
            doc["quality_status"] == "ok"
            for doc in documents
        )
    )

    print(
        "Documents avec alertes :",
        sum(
            doc["quality_status"] != "ok"
            for doc in documents
        )
    )

    print("\nInventaire sauvegardé :", OUTPUT)


if __name__ == "__main__":
    main()
