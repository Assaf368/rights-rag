"""The corpus manifest: one record per source page, whatever its file format.

The manifest is built from docs/sources.csv (id, topic, title, url) and the
saved files. PDFs carry no URL or update date, so the manifest is where every
document gets them, matched by page title.

    uv run python -m rights_rag.corpus   # writes docs/manifest.json
"""

import csv
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from rights_rag.html_to_docx import docx_name
from rights_rag.loaders.common import clean_title
from rights_rag.loaders.docx import load_docx
from rights_rag.loaders.pdf import load_pdf

DOCS = Path("docs")
MANIFEST = DOCS / "manifest.json"
LOADERS = {"pdf": load_pdf, "docx": load_docx}

_HEBREW_MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "ינואר",
            "פברואר",
            "מרץ",
            "אפריל",
            "מאי",
            "יוני",
            "יולי",
            "אוגוסט",
            "ספטמבר",
            "אוקטובר",
            "נובמבר",
            "דצמבר",
        ],
        1,
    )
}
_UPDATED = re.compile(r"עודכן לאחרונה ב־[\d:]+, (\d{1,2}) ב(\S+) (\d{4})")
_REVISION = re.compile(r'"wgRevisionId":(\d+)')


def _html_info(path: Path) -> dict:
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    og_title = soup.find("meta", property="og:title")
    updated = _UPDATED.search(html)
    revision = _REVISION.search(html)
    return {
        "title": clean_title(
            og_title["content"] if og_title else soup.title.get_text()
        ),
        "updated": (
            f"{updated[3]}-{_HEBREW_MONTHS[updated[2]]:02d}-{int(updated[1]):02d}"
            if updated and updated[2] in _HEBREW_MONTHS
            else None
        ),
        "revision_id": int(revision[1]) if revision else None,
        "saved_at": datetime.fromtimestamp(os.path.getmtime(path), UTC)
        .date()
        .isoformat(),
    }


def _pdf_info(path: Path) -> dict:
    with pymupdf.open(path) as doc:
        meta = doc.metadata
    created = re.match(r"D:(\d{4})(\d{2})(\d{2})", meta.get("creationDate") or "")
    return {
        "title": clean_title(meta.get("title") or path.stem),
        "updated": None,  # Chrome's print does not include the page's update date
        "revision_id": None,
        "saved_at": "-".join(created.groups()) if created else None,
    }


def build_manifest(docs_dir: Path = DOCS) -> list[dict]:
    sources = list(csv.DictReader((docs_dir / "sources.csv").open(encoding="utf-8")))
    found: dict[str, dict] = {}
    for pdf in sorted((docs_dir / "pdf").glob("*.pdf")):
        info = _pdf_info(pdf)
        found[info.pop("title")] = {"path": str(pdf), "source_html": None, **info}
    for html in sorted((docs_dir / "html").glob("*.html")):
        info = _html_info(html)
        docx = docs_dir / "docx" / docx_name(html)
        found[info.pop("title")] = {"path": str(docx), "source_html": str(html), **info}

    records, problems = [], []
    for src in sources:
        hit = found.pop(src["title"], None)
        if hit is None:
            problems.append(f"missing file for {src['id']} {src['title']}")
            continue
        fmt = Path(hit["path"]).suffix.lstrip(".")
        if fmt != src["format"]:
            problems.append(
                f"{src['id']} {src['title']}: saved as {fmt}, listed as {src['format']}"
            )
        records.append({**src, "format": fmt, **hit})
    problems += [f"file not in sources.csv: {title}" for title in found]
    if problems:
        raise ValueError("manifest problems:\n" + "\n".join(problems))
    return records


def load_corpus(manifest: Path = MANIFEST) -> list[Document]:
    """Load every manifest entry with its format's loader and attach the
    manifest fields (id, url, topic, updated, revision) to each section."""
    docs = []
    for rec in json.loads(manifest.read_text(encoding="utf-8")):
        for doc in LOADERS[rec["format"]](rec["path"]):
            doc.metadata |= {
                "doc_id": rec["id"],
                "url": rec["url"],
                "topic": rec["topic"],
                "updated": rec["updated"],
                "revision_id": rec["revision_id"],
            }
            docs.append(doc)
    return docs


def main() -> None:
    records = build_manifest()
    MANIFEST.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    by_format = {f: sum(r["format"] == f for r in records) for f in LOADERS}
    print(f"wrote {MANIFEST}: {len(records)} records {by_format}")


if __name__ == "__main__":
    main()
