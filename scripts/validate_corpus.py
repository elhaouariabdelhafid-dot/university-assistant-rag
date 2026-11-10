
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INVENTORY = ROOT / "data/metadata/documents.csv"
CHUNKS_FILE = ROOT / "artifacts/chunks.json"
TEXT_DIR = ROOT / "data/processed/text"


def main():
    print("=" * 65)
    print("VALIDATION DU CORPUS ET DE LA PROVENANCE")
    print("=" * 65)

    errors = []

    # 1. Charger l'inventaire
    with INVENTORY.open(encoding="utf-8", newline="") as f:
        documents = list(csv.DictReader(f))

    # 2. Vérifier l'intégrité SHA-256
    source_names = set()
    document_ids = set()

    for doc in documents:
        path = ROOT / doc["relative_path"]

        if not path.is_file():
            errors.append(f"Document manquant : {path}")
            continue

        actual_hash = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

        if actual_hash != doc["sha256"]:
            errors.append(
                f"SHA-256 incorrect : {doc['filename']}"
            )

        source_names.add(path.stem)

        if doc["document_id"] in document_ids:
            errors.append(
                f"Identifiant dupliqué : {doc['document_id']}"
            )

        document_ids.add(doc["document_id"])

    # 3. Vérifier que tous les TXT sont inventoriés
    actual_files = {
        path.stem
        for path in TEXT_DIR.glob("DOC*.txt")
    }

    for source in sorted(actual_files - source_names):
        errors.append(
            f"Document absent de l'inventaire : {source}"
        )

    # 4. Vérifier la provenance des chunks
    chunks = json.loads(
        CHUNKS_FILE.read_text(encoding="utf-8")
    )

    chunk_counts = Counter()
    seen_chunks = set()

    for chunk in chunks:
        source = chunk["source"]
        chunk_id = int(chunk["chunk_id"])

        if source not in source_names:
            errors.append(
                f"Source inconnue : {source}"
            )

        key = (source, chunk_id)

        if key in seen_chunks:
            errors.append(
                f"Chunk dupliqué : {key}"
            )

        seen_chunks.add(key)

        if not chunk["text"].strip():
            errors.append(
                f"Chunk vide : {key}"
            )

        chunk_counts[source] += 1

    for source in sorted(source_names):
        if chunk_counts[source] == 0:
            errors.append(
                f"Aucun chunk pour : {source}"
            )

    # 5. Afficher les résultats
    print("\nDOCUMENTS :", len(documents))
    print("CHUNKS :", len(chunks))

    print("\nCHUNKS PAR DOCUMENT :")

    for source in sorted(source_names):
        print(
            f"{source} : {chunk_counts[source]} chunks"
        )

    print("\n" + "=" * 65)

    if errors:
        print("VALIDATION : ÉCHEC")

        for error in errors:
            print("[ERREUR]", error)

        raise SystemExit(1)

    print("VALIDATION : RÉUSSIE")
    print("Intégrité SHA-256 : OK")
    print("Traçabilité des chunks : OK")
    print("Identifiants : OK")


if __name__ == "__main__":
    main()
