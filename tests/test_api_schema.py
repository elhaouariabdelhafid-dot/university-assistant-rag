from src.api.main import Question

def test_question_validation():
    assert Question(question="Où trouver le règlement ?").question
