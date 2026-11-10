
import time

from src.rag.pipeline import UniversityRAG


TEST_CASES = [
    {
        "question": "Quels documents faut-il pour une première inscription ?",
        "expected_refused": False,
        "expected_source": "DOC02_inscription_reinscription"
    },
    {
        "question": "Comment se déroule la réinscription des étudiants ?",
        "expected_refused": False,
        "expected_source": "DOC02_inscription_reinscription"
    },
    {
        "question": "Quel est le salaire du président de l'Université Al Amal ?",
        "expected_refused": True,
        "expected_source": None
    },
    {
        "question": "Quelle est la couleur du bâtiment principal de l'université ?",
        "expected_refused": True,
        "expected_source": None
    }
]


def main():

    print("=" * 70)
    print("TEST DE GÉNÉRATION - UNIVERSITY ASSISTANT RAG")
    print("=" * 70)

    rag = UniversityRAG()

    passed = 0
    latencies = []

    for i, test in enumerate(TEST_CASES, 1):

        question = test["question"]

        start = time.perf_counter()
        result = rag.answer(question)
        elapsed = time.perf_counter() - start

        actual_sources = [
            source["source"]
            for source in result["sources"]
        ]

        refusal_ok = (
            result["refused"] == test["expected_refused"]
        )

        expected_source = test["expected_source"]

        source_ok = (
            expected_source in actual_sources
            if expected_source
            else len(actual_sources) == 0
        )

        success = refusal_ok and source_ok

        if success:
            passed += 1

        latencies.append(elapsed)

        print("\n" + "=" * 70)
        print(f"TEST {i} : {question}")

        print("\nRÉPONSE :")
        print(result["answer"])

        print("\nMODÈLE :", result["model"])
        print("REFUSED :", result["refused"])
        print("SOURCES :", actual_sources)

        print(
            f"RETRIEVAL : "
            f"{result['retrieval_s']:.3f} s"
        )

        print(
            f"GÉNÉRATION : "
            f"{result['latency_s']:.3f} s"
        )

        print(
            f"TEMPS TOTAL : "
            f"{result['total_latency_s']:.3f} s"
        )

        print(
            "STATUT :", "PASS" if success else "FAIL"
        )

    print("\n" + "=" * 70)
    print("RÉSULTATS FINAUX")
    print("=" * 70)

    print(f"Tests réussis : {passed}/{len(TEST_CASES)}")

    mean_latency = sum(latencies) / len(latencies)

    print(
        f"Latence moyenne : {mean_latency:.3f} s"
    )

    if passed == len(TEST_CASES):
        print("VALIDATION FONCTIONNELLE RÉUSSIE")
    else:
        print("CERTAINS TESTS ONT ÉCHOUÉ")

    # Statut de sortie pour CI/CD
    if passed != len(TEST_CASES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
