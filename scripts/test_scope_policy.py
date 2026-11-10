
from src.rag.scope_policy import check_scope


TEST_CASES = [
    (
        "Comment obtenir une attestation de scolarité ?",
        True
    ),
    (
        "Comment demander mon relevé de notes ?",
        True
    ),
    (
        "Quels documents faut-il pour une inscription ?",
        True
    ),
    (
        "Quel est le salaire du président ?",
        True
    ),
    (
        "Comment modifier mon inscription ?",
        True
    ),
    (
        "Donne-moi les notes d'un étudiant.",
        False
    ),
    (
        "Modifie mon inscription.",
        False
    ),
    (
        "Quel médicament dois-je prendre ?",
        False
    ),
    (
        "Quel est le prix du Bitcoin aujourd'hui ?",
        False
    ),
    (
        "Quel traitement médical dois-je prendre ?",
        False
    )
]


def main():

    passed = 0

    print("=" * 65)
    print("TESTS DE PÉRIMÈTRE - UNIVERSITY ASSISTANT RAG")
    print("=" * 65)

    for i, (question, expected) in enumerate(
        TEST_CASES, 1
    ):

        result = check_scope(question)

        success = result["allowed"] == expected

        if success:
            passed += 1

        print(f"\nTEST {i} : {question}")
        print("Autorisé :", result["allowed"])
        print("Catégorie :", result["category"])
        print("STATUT :", "PASS" if success else "FAIL")

    print("\n" + "=" * 65)
    print(f"RÉSULTAT : {passed}/{len(TEST_CASES)}")
    print("=" * 65)

    if passed != len(TEST_CASES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
