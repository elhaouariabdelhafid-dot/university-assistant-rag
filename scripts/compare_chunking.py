
from pathlib import Path

from src.rag.chunking import split_words


ROOT = Path(__file__).resolve().parents[1]
TEXT_DIR = ROOT / "data" / "processed" / "text"

CONFIGS = {
    "A_reference": {
        "size": 180,
        "overlap": 35,
    },
    "B_experimentale": {
        "size": 120,
        "overlap": 20,
    },
}


def main():
    files = sorted(TEXT_DIR.glob("*.txt"))

    if not files:
        raise FileNotFoundError(
            f"Aucun document TXT trouvé dans {TEXT_DIR}"
        )

    print("=" * 72)
    print("ÉTAPE 6.1 - COMPARAISON DES CONFIGURATIONS DE CHUNKING")
    print("=" * 72)
    print(f"Nombre de documents : {len(files)}")

    results = {}

    for config_name, params in CONFIGS.items():
        print("\n" + "-" * 72)
        print(f"CONFIGURATION : {config_name}")
        print(f"Taille : {params['size']} mots")
        print(f"Overlap : {params['overlap']} mots")
        print("-" * 72)

        total_chunks = 0
        total_words = 0
        chunk_lengths = []

        for path in files:
            text = path.read_text(encoding="utf-8")

            chunks = split_words(
                text,
                size=params["size"],
                overlap=params["overlap"],
            )

            document_words = len(text.split())
            total_words += document_words
            total_chunks += len(chunks)

            lengths = [len(chunk.split()) for chunk in chunks]
            chunk_lengths.extend(lengths)

            print(
                f"{path.stem:<38} "
                f"{len(chunks):>3} chunks"
            )

        mean_length = (
            sum(chunk_lengths) / len(chunk_lengths)
            if chunk_lengths else 0
        )

        results[config_name] = {
            "chunks": total_chunks,
            "words": total_words,
            "mean_length": mean_length,
        }

        print(f"\nTOTAL CHUNKS : {total_chunks}")
        print(f"TOTAL MOTS SOURCE : {total_words}")
        print(f"TAILLE MOYENNE CHUNK : {mean_length:.2f} mots")

    print("\n" + "=" * 72)
    print("SYNTHÈSE")
    print("=" * 72)

    for name, result in results.items():
        print(
            f"{name}: "
            f"{result['chunks']} chunks, "
            f"taille moyenne {result['mean_length']:.2f} mots"
        )

    print("\nAucun index FAISS existant n'a été modifié.")


if __name__ == "__main__":
    main()
