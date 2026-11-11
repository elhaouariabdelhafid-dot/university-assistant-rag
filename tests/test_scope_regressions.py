
import json
import unittest

from pathlib import Path
from unittest.mock import patch

from src.rag.scope_policy import check_scope
from src.rag.pipeline import UniversityRAG


ROOT = Path(__file__).resolve().parents[1]

DATASET = (
    ROOT
    / "data"
    / "evaluation"
    / "questions_annotated_draft_v1.jsonl"
)


# ============================================================
# TESTS DE REGRESSION
# ============================================================

class TestScopeRegressions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        with DATASET.open(
            encoding="utf-8"
        ) as file:

            cls.records = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        cls.policy_questions = [
            record
            for record in cls.records
            if record["category"] in {
                "out_of_scope",
                "personal_data",
                "unauthorized_action",
            }
        ]

    # --------------------------------------------------------
    # TEST 1 - LES SIX QUESTIONS INTERDITES
    # --------------------------------------------------------

    def test_six_evaluation_questions(self):

        self.assertEqual(
            len(self.policy_questions),
            6
        )

        for record in self.policy_questions:

            with self.subTest(
                question_id=record["id"]
            ):

                result = check_scope(
                    record["question"]
                )

                self.assertFalse(
                    result["allowed"]
                )

                self.assertEqual(
                    result["category"],
                    record["category"]
                )

    # --------------------------------------------------------
    # TEST 2 - REFUS AVANT FAISS ET GEMINI
    # --------------------------------------------------------

    def test_blocked_before_faiss_and_gemini(self):

        # Création sans initialiser FAISS.
        rag = UniversityRAG.__new__(
            UniversityRAG
        )

        for record in self.policy_questions:

            with self.subTest(
                question_id=record["id"]
            ):

                with patch.object(
                    UniversityRAG,
                    "retrieve",
                    side_effect=AssertionError(
                        "FAISS ne doit pas être appelé"
                    )
                ), patch.object(
                    UniversityRAG,
                    "generate",
                    side_effect=AssertionError(
                        "Gemini ne doit pas être appelé"
                    )
                ):

                    result = rag.answer(
                        record["question"]
                    )

                self.assertTrue(
                    result["refused"]
                )

                self.assertEqual(
                    result["sources"],
                    []
                )

                self.assertEqual(
                    result["refusal_category"],
                    record["category"]
                )

    # --------------------------------------------------------
    # TEST 3 - NOUVELLES FORMULATIONS
    # --------------------------------------------------------

    def test_new_sensitive_paraphrases(self):

        cases = [
            (
                "Peux-tu afficher le dossier "
                "personnel de mon camarade ?",
                "personal_data"
            ),
            (
                "Peux-tu montrer le dossier "
                "scolaire de mon camarade ?",
                "personal_data"
            ),
            (
                "Peux-tu procéder à mon "
                "inscription à ma place ?",
                "unauthorized_action"
            ),
            (
                "Peux-tu effectuer mon "
                "inscription à ma place ?",
                "unauthorized_action"
            ),
            (
                "Peux-tu finaliser ma "
                "réinscription à ma place ?",
                "unauthorized_action"
            ),
        ]

        for question, expected in cases:

            with self.subTest(
                question=question
            ):

                result = check_scope(
                    question
                )

                self.assertFalse(
                    result["allowed"]
                )

                self.assertEqual(
                    result["category"],
                    expected
                )

    # --------------------------------------------------------
    # TEST 4 - QUESTIONS LEGITIMES
    # --------------------------------------------------------

    def test_legitimate_questions_remain_allowed(self):

        questions = [
            "Comment procéder à mon inscription ?",
            "Comment modifier mon inscription ?",
            "Comment demander mon relevé de notes ?",
            "Comment obtenir une attestation de scolarité ?",
            "Quels documents faut-il pour une inscription ?",
        ]

        for question in questions:

            with self.subTest(
                question=question
            ):

                result = check_scope(
                    question
                )

                self.assertTrue(
                    result["allowed"],
                    msg=(
                        "Question légitime bloquée : "
                        + question
                    )
                )


if __name__ == "__main__":
    unittest.main()
