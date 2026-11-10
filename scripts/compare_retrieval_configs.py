
from pathlib import Path
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.rag.chunking import split_words


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
TEXT_DIR = ROOT / "data" / "processed" / "text"

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

TOP_K = 5

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


# Jeu de questions préliminaire.
# Les documents attendus devront être vérifiés
# manuellement lors de l'étape 7.

QUESTIONS = [
    {
        "question": (
            "Quelle est la mission de "
            "l'Université Al Amal ?"
        ),
        "expected_source": "DOC01_presentation_universite",
    },
    {
        "question": (
            "Quels documents faut-il pour "
            "une première inscription ?"
        ),
        "expected_source": "DOC02_inscription_reinscription",
    },
    {
        "question": (
            "Quelles sont les règles "
            "des examens universitaires ?"
        ),
        "expected_source": "DOC03_reglement_examens",
    },
    {
        "question": (
            "Comment justifier une absence "
            "à l'université ?"
        ),
        "expected_source": "DOC04_absences_et_justifications",
    },
    {
        "question": "Quelles sont les étapes du PFE ?",
        "expected_source": "DOC05_stages_et_PFE",
    },
    {
        "question": (
            "Quels programmes de formation "
            "propose l'université ?"
        ),
        "expected_source": "DOC06_programmes_formations",
    },
    {
        "question": (
            "Quelles sont les principales "
            "périodes du calendrier universitaire ?"
        ),
        "expected_source": "DOC07_calendrier_universitaire",
    },
    {
        "question": (
            "Comment obtenir une attestation "
            "de scolarité ?"
        ),
        "expected_source": "DOC08_frais_procedures",
    },
]


# ============================================================
# PRÉPARATION DU CORPUS
# ============================================================

def prepare_chunks(files, size, overlap):

    texts = []
    sources = []

    for path in files:

        content = path.read_text(encoding="utf-8")

        chunks = split_words(
            content,
            size=size,
            overlap=overlap,
        )

        for chunk in chunks:
            texts.append(chunk)
            sources.append(path.stem)

    return texts, sources


# ============================================================
# CONSTRUCTION DE L'INDEX
# ============================================================

def build_index(model, texts):

    start = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    vectors = np.ascontiguousarray(
        embeddings,
        dtype=np.float32,
    )

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    build_time = time.perf_counter() - start

    return index, build_time


# ============================================================
# MÉTRIQUES RETRIEVAL
# ============================================================

def evaluate(index, sources, question_vectors):

    hit_at_1 = 0
    hit_at_5 = 0
    reciprocal_ranks = []

    details = []

    for case, vector in zip(
        QUESTIONS,
        question_vectors,
    ):

        query_vector = np.ascontiguousarray(
            vector.reshape(1, -1),
            dtype=np.float32,
        )

        scores, ids = index.search(
            query_vector,
            TOP_K,
        )

        expected = case["expected_source"]

        rank = None
        first_source = None

        for position, chunk_id in enumerate(
            ids[0],
            start=1,
        ):

            if chunk_id < 0:
                continue

            source = sources[int(chunk_id)]

            if first_source is None:
                first_source = source

            if source == expected and rank is None:
                rank = position

        if rank == 1:
            hit_at_1 += 1

        if rank is not None:
            hit_at_5 += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        details.append({
            "question": case["question"],
            "expected": expected,
            "first_source": first_source,
            "rank": rank,
        })

    count = len(QUESTIONS)

    return {
        "hit_at_1": hit_at_1 / count,
        "recall_at_5": hit_at_5 / count,
        "mrr_at_5": sum(reciprocal_ranks) / count,
        "details": details,
    }


# ============================================================
# MESURE DE LATENCE FAISS
# ============================================================

def measure_faiss_latency(
    index,
    question_vectors,
    repetitions=100,
):

    # Échauffement avant la mesure.
    for vector in question_vectors:

        query = np.ascontiguousarray(
            vector.reshape(1, -1),
            dtype=np.float32,
        )

        index.search(query, TOP_K)

    start = time.perf_counter()

    total_searches = 0

    for _ in range(repetitions):

        for vector in question_vectors:

            query = np.ascontiguousarray(
                vector.reshape(1, -1),
                dtype=np.float32,
            )

            index.search(query, TOP_K)
            total_searches += 1

    elapsed = time.perf_counter() - start

    return (elapsed / total_searches) * 1000


# ============================================================
# AFFICHAGE
# ============================================================

def print_results(name, chunk_count, build_time, metrics):

    print("\n" + "=" * 72)
    print(f"CONFIGURATION : {name}")
    print("=" * 72)

    print(f"Nombre de chunks : {chunk_count}")
    print(f"Temps de construction : {build_time:.3f} s")

    print(f"Hit@1 : {metrics['hit_at_1']:.2%}")
    print(f"Recall@5 : {metrics['recall_at_5']:.2%}")
    print(f"MRR@5 : {metrics['mrr_at_5']:.4f}")

    print(
        "Latence FAISS moyenne : "
        f"{metrics['latency_ms']:.4f} ms"
    )

    print("\nDÉTAIL DES QUESTIONS :")

    for i, item in enumerate(
        metrics["details"],
        start=1,
    ):

        rank = (
            str(item["rank"])
            if item["rank"] is not None
            else "ABSENT"
        )

        print(f"\nQUESTION {i}")
        print(f"  {item['question']}")
        print(f"  Document attendu : {item['expected']}")
        print(f"  Premier document : {item['first_source']}")
        print(f"  Rang attendu : {rank}")


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    faiss.omp_set_num_threads(1)

    files = sorted(TEXT_DIR.glob("*.txt"))

    if not files:
        raise FileNotFoundError(
            f"Aucun fichier TXT dans {TEXT_DIR}"
        )

    corpus_sources = {path.stem for path in files}

    for case in QUESTIONS:
        if case["expected_source"] not in corpus_sources:
            raise ValueError(
                "Document attendu introuvable : "
                + case["expected_source"]
            )

    print("=" * 72)
    print("ÉTAPE 6.2 - COMPARAISON DU RETRIEVAL")
    print("=" * 72)

    print(f"Documents : {len(files)}")
    print(f"Questions : {len(QUESTIONS)}")
    print(f"Modèle : {EMBEDDING_MODEL}")
    print(f"TOP_K : {TOP_K}")

    print("\nChargement du modèle d'embeddings...")

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu",
    )

    # Les embeddings des questions sont calculés
    # une seule fois, puis réutilisés pour A et B.
    question_vectors = model.encode(
        [case["question"] for case in QUESTIONS],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    question_vectors = np.ascontiguousarray(
        question_vectors,
        dtype=np.float32,
    )

    all_results = {}

    for name, params in CONFIGS.items():

        print(f"\nConstruction de {name}...")

        texts, sources = prepare_chunks(
            files,
            size=params["size"],
            overlap=params["overlap"],
        )

        index, build_time = build_index(
            model,
            texts,
        )

        if index.ntotal != len(sources):
            raise AssertionError(
                "Incohérence entre index et chunks."
            )

        metrics = evaluate(
            index,
            sources,
            question_vectors,
        )

        metrics["latency_ms"] = measure_faiss_latency(
            index,
            question_vectors,
        )

        print_results(
            name,
            len(texts),
            build_time,
            metrics,
        )

        all_results[name] = {
            "chunks": len(texts),
            **metrics,
        }

    print("\n" + "=" * 72)
    print("COMPARAISON FINALE")
    print("=" * 72)

    print(
        f"{'Métrique':<24}"
        f"{'A_reference':>18}"
        f"{'B_experimentale':>20}"
    )

    keys = [
        ("Chunks", "chunks"),
        ("Hit@1", "hit_at_1"),
        ("Recall@5", "recall_at_5"),
        ("MRR@5", "mrr_at_5"),
        ("Latence FAISS (ms)", "latency_ms"),
    ]

    a = all_results["A_reference"]
    b = all_results["B_experimentale"]

    for label, key in keys:

        a_value = a[key]
        b_value = b[key]

        if key == "chunks":
            left = str(a_value)
            right = str(b_value)

        elif key == "latency_ms":
            left = f"{a_value:.4f}"
            right = f"{b_value:.4f}"

        else:
            left = f"{a_value:.4f}"
            right = f"{b_value:.4f}"

        print(
            f"{label:<24}"
            f"{left:>18}"
            f"{right:>20}"
        )

    print("\nIMPORTANT :")
    print("- Métriques évaluées au niveau du document.")
    print("- Jeu de 8 questions préliminaire.")
    print("- Latence limitée à la recherche FAISS.")
    print("- Aucun appel Gemini.")
    print("- Aucun fichier d'index existant modifié.")


if __name__ == "__main__":
    main()
