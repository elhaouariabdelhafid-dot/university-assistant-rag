"""Évaluation minimale. Compléter data/evaluation/questions.json avec des cas annotés."""
from pathlib import Path
import json
from src.rag.pipeline import UniversityRAG

ROOT = Path(__file__).resolve().parents[1]
dataset = ROOT / "data/evaluation/questions.json"
if not dataset.exists():
    raise SystemExit("Créez data/evaluation/questions.json à partir du modèle fourni.")
cases = json.loads(dataset.read_text(encoding="utf-8"))
rag = UniversityRAG()
hits = 0
for case in cases:
    retrieved = rag.retrieve(case["question"])
    sources = {x["source"] for x in retrieved}
    expected = set(case.get("expected_sources", []))
    hit = bool(sources & expected) if expected else False
    hits += int(hit)
    print(f"{case['question']} -> sources={sorted(sources)} | hit={hit}")
print(f"Recall source-level (sur cas avec source attendue) : {hits}/{len(cases)}")
