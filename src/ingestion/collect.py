from pathlib import Path
import hashlib
import requests
import yaml
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
HEADERS = {"User-Agent": "UniversityAssistantAcademicPrototype/1.0 (document collection)"}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def collect():
    config = yaml.safe_load((ROOT / "configs/sources.yaml").read_text(encoding="utf-8"))
    inventory = []
    for source in config["sources"]:
        folder = RAW / ("pdfs" if source["type"] == "pdf" else "html")
        folder.mkdir(parents=True, exist_ok=True)
        ext = ".pdf" if source["type"] == "pdf" else ".html"
        path = folder / f'{source["id"]}{ext}'
        response = requests.get(source["url"], headers=HEADERS, timeout=30)
        response.raise_for_status()
        content = response.content
        if source["type"] == "pdf" and not content.startswith(b"%PDF"):
            raise ValueError(f"Le fichier reçu pour {source['id']} n'est pas un PDF valide.")
        path.write_bytes(content)
        inventory.append({
            **source, "local_path": str(path.relative_to(ROOT)),
            "sha256": sha256(path), "bytes": len(content)
        })
        print(f"[OK] {source['id']} -> {path}")
    import csv
    out = ROOT / "data/metadata"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "documents.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in inventory for k in row}))
        writer.writeheader()
        writer.writerows(inventory)

if __name__ == "__main__":
    collect()
