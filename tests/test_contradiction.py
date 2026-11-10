
import unittest
from pathlib import Path

from src.rag.contradiction import detect_contradictions


ROOT = Path(__file__).resolve().parents[1]

FIXTURES = (
    ROOT / "tests" / "fixtures" / "contradictions"
)

QUESTION = (
    "Quel est le délai pour justifier une absence ?"
)


def load_document(filename):

    path = FIXTURES / filename

    return {
        "source": path.stem,
        "chunk_id": 0,
        "text": path.read_text(encoding="utf-8")
    }


class TestContradictions(unittest.TestCase):

    def setUp(self):

        self.doc_a = load_document("DOC_A.txt")
        self.doc_b = load_document("DOC_B.txt")

    def test_conflicting_deadlines(self):

        result = detect_contradictions(
            QUESTION,
            [self.doc_a, self.doc_b]
        )

        self.assertTrue(result["has_conflict"])

        self.assertEqual(
            result["category"],
            "conflicting_deadlines"
        )

        hours = {
            item["hours"]
            for item in result["evidence"]
        }

        self.assertEqual(hours, {48, 72})

    def test_single_document(self):

        result = detect_contradictions(
            QUESTION,
            [self.doc_a]
        )

        self.assertFalse(result["has_conflict"])

    def test_identical_deadlines(self):

        doc_c = {
            **self.doc_a,
            "source": "DOC_C"
        }

        result = detect_contradictions(
            QUESTION,
            [self.doc_a, doc_c]
        )

        self.assertFalse(result["has_conflict"])

    def test_unrelated_question(self):

        result = detect_contradictions(
            "Quels sont les programmes de formation ?",
            [self.doc_a, self.doc_b]
        )

        self.assertFalse(result["has_conflict"])

    def test_unrelated_deadline(self):

        doc_c = {
            "source": "DOC_C",
            "chunk_id": 0,
            "text": (
                "Le relevé de notes est disponible "
                "dans un délai de 48 heures après "
                "la publication des résultats."
            )
        }

        result = detect_contradictions(
            QUESTION,
            [self.doc_a, doc_c]
        )

        self.assertFalse(result["has_conflict"])


if __name__ == "__main__":
    unittest.main()
