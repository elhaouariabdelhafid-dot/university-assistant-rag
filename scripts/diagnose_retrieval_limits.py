
from src.rag.pipeline import UniversityRAG, MIN_SCORE


TEST_CASES = [
    {
        "question": "Quels documents faut-il pour une première inscription ?",
        "category": "in_scope"
    },
    {
        "question": "Comment obtenir une attestation de scolarité ?",
        "category": "in_scope"
    },
    {
        "question": "Quelles sont les étapes du PFE ?",
        "category": "in_scope"
    },
    {
        "question": "Quel est le salaire du président de l'Université Al Amal ?",
        "category": "out_of_corpus"
    },
    {
        "question": "Quelle est la couleur du bâtiment principal ?",
        "category": "out_of_corpus"
    },
    {
        "question": "Quel est le prix du Bitcoin aujourd'hui ?",
        "category": "out_of_scope"
    },
    {
        "question": "Quel traitement médical dois-je prendre ?",
        "category": "out_of_scope"
    }
]


def main():

    print("=" * 70)
    print("ÉTAPE 5.1 - DIAGNOSTIC DES LIMITES DU RETRIEVAL")
    print("=" * 70)

    rag = UniversityRAG()

    print(f"\nSEUIL ACTUEL : {MIN_SCORE}")

    for index, test in enumerate(TEST_CASES, 1):

        question = test["question"]

        docs = rag.retrieve(question)

        print("\n" + "-" * 70)
        print(f"TEST {index}/{len(TEST_CASES)}")
        print("CATÉGORIE :", test["category"])
        print("QUESTION :", question)

        if not docs:
            print("Aucun document retrouvé.")
            print("DÉCISION DU SEUIL : REFUS")
            continue

        best = docs[0]
        score = best["score"]

        print("MEILLEURE SOURCE :", best["source"])
        print("MEILLEUR CHUNK :", best["chunk_id"])
        print(f"SCORE FAISS : {score:.4f}")

        threshold_passed = score >= MIN_SCORE

        print(
            "DÉCISION DU SEUIL :",
            "ACCEPTER LE CONTEXTE"
            if threshold_passed
            else "REFUSER"
        )

        print("\nTOP 3 :")

        for doc in docs[:3]:
            print(
                f"  {doc['source']} "
                f"| chunk={doc['chunk_id']} "
                f"| score={doc['score']:.4f}"
            )

        if test["category"] != "in_scope" and threshold_passed:
            print(
                "\n[ATTENTION] Le seuil accepte du contexte "
                "pour une question non couverte."
            )
            print(
                "Cela ne prouve pas que le LLM répondra "
                "incorrectement, mais révèle une limite "
                "du filtre de retrieval."
            )

    print("\n" + "=" * 70)
    print("DIAGNOSTIC TERMINÉ - AUCUN APPEL GEMINI")
    print("=" * 70)


if __name__ == "__main__":
    main()
