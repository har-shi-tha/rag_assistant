from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)


def load_documents(folder: str) -> List[Dict]:
    """
    Loads .txt/.md/.pdf from a folder recursively.
    Returns: [{"source": "...", "text": "..."}]
    """
    base = Path(folder)
    docs: List[Dict] = []

    if not base.exists():
        return docs

    for path in base.rglob("*"):
        if path.is_dir():
            continue

        suffix = path.suffix.lower()
        if suffix in [".txt", ".md"]:
            text = load_txt(path)
        elif suffix == ".pdf":
            text = load_pdf(path)
        else:
            continue

        text = (text or "").strip()
        if text:
            docs.append({"source": str(path), "text": text})

    return docs