from src.rag.chunking import split_words

def test_chunking_non_empty():
    chunks = split_words(" ".join(["mot"] * 500), size=100, overlap=20)
    assert len(chunks) > 1
    assert all(chunk for chunk in chunks)

def test_invalid_chunking_parameters():
    import pytest
    with pytest.raises(ValueError):
        split_words("texte", size=10, overlap=10)
