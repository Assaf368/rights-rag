"""Load Word documents into the same shape as the PDF loader.

Word keeps structure explicitly, so nothing is guessed from layout: heading
styles give the section path, list styles give bullets, and a table's first
row is its header when Word marks it as a repeating header row.
"""

from pathlib import Path

import docx
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain_core.documents import Document

from rights_rag.loaders.common import Section, clean_title, format_rows, to_documents

H2_STYLE = "Heading 1"
H3_STYLE = "Heading 2"
LIST_STYLE_PREFIXES = ("List Bullet", "List Number")


def _paragraph_text(p: Paragraph) -> str:
    text = " ".join(p.text.split())
    if text and p.style.name.startswith(LIST_STYLE_PREFIXES):
        return "• " + text
    return text


def _cell_text(cell) -> str:
    return " ".join(t for t in (_paragraph_text(p) for p in cell.paragraphs) if t)


def _table_rows(table: Table) -> list[str]:
    rows = []
    for row in table.rows:
        # Merged cells repeat the same cell object; keep each once.
        cells, seen = [], set()
        for cell in row.cells:
            if id(cell._tc) not in seen:
                seen.add(id(cell._tc))
                cells.append(_cell_text(cell))
        rows.append(cells)
    if not rows:
        return []
    first = table.rows[0]._tr
    has_header = (
        first.trPr is not None and first.trPr.find(qn("w:tblHeader")) is not None
    )
    if has_header:
        return format_rows(rows[0], rows[1:])
    return format_rows(None, rows)


def load_docx(path: str | Path) -> list[Document]:
    """Return one Document per section, with the heading path in metadata."""
    path = Path(path)
    doc = docx.Document(path)
    title = clean_title(doc.core_properties.title or path.stem)

    sections: list[Section] = []
    h2 = h3 = None
    for item in doc.iter_inner_content():
        if isinstance(item, Table):
            texts = _table_rows(item)
        else:
            style = item.style.name
            if style == "Title":
                continue
            if style in (H2_STYLE, H3_STYLE):
                if style == H2_STYLE:
                    h2, h3 = item.text.strip(), None
                else:
                    h3 = item.text.strip()
                sections.append(Section(h2, h3))
                continue
            texts = [t for t in [_paragraph_text(item)] if t]
        if not texts:
            continue
        if not sections:
            sections.append(Section(None, None))
        sections[-1].texts.extend(texts)

    return to_documents(sections, title=title, source=str(path), fmt="docx")
