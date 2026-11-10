from pathlib import Path
import re

def split_words(text: str, size: int = 180, overlap: int = 35):
    words = re.findall(r"\S+", text)
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Paramètres invalides : size > overlap >= 0.")
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start:start + size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + size >= len(words):
            break
    return chunks

def load_chunks(text_dir: Path, size=180, overlap=35):
    records = []
    for path in sorted(text_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(split_words(text, size, overlap)):
            records.append({
                "text": chunk, "source": path.stem, "chunk_id": i,
                "status": "unknown"
            })
    return records
