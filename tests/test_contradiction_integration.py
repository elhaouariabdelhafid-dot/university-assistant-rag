
import unittest
from pathlib import Path
from unittest.mock import patch

from src.rag.pipeline import UniversityRAG


ROOT = Path(__file__).resolve().parents[1]

FIXTURES = (
    ROOT / "tests" / "fixtures" / "contradictions"
)

QUESTION = "Quel est le délai pour justifier une absence ?"


def load_doc(filename, score):

    path = FIXTURES / filename

    return {
        "source": path.stem,
        "chunk_id": 0,
        "text": path.read_text(encoding="utf-8"),
        "score": score
    }


class TestContradictionIntegration(unittest.TestCase):

    def setUp(self):

        # Éviter de charger FAISS et les embeddings
        self.rag = UniversityRAG.__new__(UniversityRAG)

        self.doc_a = load_doc("DOC_A.txt", 0.80)
        self.doc_b = load_doc("DOC_B.txt", 0.75)

    def test_conflict_blocks_generation(self):

        documents = [self.doc_a, self.doc_b]

        with patch.object(
            UniversityRAG,
            "retrieve",
            return_value=documents
        ), patch.object(
            UniversityRAG,
            "expand_context",
            return_value=documents
        ), patch.object(
            UniversityRAG,
            "generate",
            side_effect=AssertionError(
                "Gemini ne doit pas être appelé"
            )
        ) as mock_generate:

            result = self.rag.answer(QUESTION)

        self.assertTrue(result["refused"])

        self.assertTrue(
            result.get("conflict_detected", False)
        )

        self.assertEqual(
            result.get("refusal_category"),
            "document_conflict"
        )

        self.assertIn("[DOC_A]", result["answer"])
        self.assertIn("[DOC_B]", result["answer"])

        source_ids = {
            item["source"]
            for item in result["sources"]
        }

        self.assertEqual(
            source_ids,
            {"DOC_A", "DOC_B"}
        )

        mock_generate.assert_not_called()

    def test_no_conflict_allows_generation(self):

        doc_b_same = self.doc_b.copy()

        doc_b_same["text"] = (
            doc_b_same["text"].replace(
                "48 heures",
                "72 heures"
            )
        )

        documents = [self.doc_a, doc_b_same]

        fake_generation = {
            "answer": (
                "Le délai est de 72 heures. [DOC_A]"
            ),
            "latency_s": 0.05,
            "model": "test-mock"
        }

        with patch.object(
            UniversityRAG,
            "retrieve",
            return_value=documents
        ), patch.object(
            UniversityRAG,
            "expand_context",
            return_value=documents
        ), patch.object(
            UniversityRAG,
            "generate",
            return_value=fake_generation
        ) as mock_generate:

            result = self.rag.answer(QUESTION)

        self.assertFalse(result["refused"])
        self.assertTrue(result["citations_present"])

        mock_generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
