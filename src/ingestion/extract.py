from pathlib import Path
import pymupdf
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed/text"

def normalize(text: str) -> str:
    lines = [line.strip() for line in text.replace("\x00", " ").splitlines()]
    return "\n".join(line for line in lines if line)

def extract_all():
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (RAW / "pdfs").glob("*.pdf"):
        doc = pymupdf.open(path)
        pages = [f"\n[PAGE {i+1}]\n{page.get_text('text')}" for i, page in enumerate(doc)]
        text = normalize("\n".join(pages))
        (OUT / f"{path.stem}.txt").write_text(text, encoding="utf-8")
        if len(text) < 100:
            print(f"[WARN] Texte court/scanné : {path.name}; OCR à envisager.")
        else:
            print(f"[OK] PDF extrait : {path.name} ({len(text)} caractères)")
    for path in (RAW / "html").glob("*.html"):
        soup = BeautifulSoup(path.read_bytes(), "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = normalize(soup.get_text("\n"))
        (OUT / f"{path.stem}.txt").write_text(text, encoding="utf-8")
        print(f"[OK] HTML extrait : {path.name} ({len(text)} caractères)")

if __name__ == "__main__":
    extract_all()
