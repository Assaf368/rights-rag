"""Chunking strategies. Each turns the loaded sections into chunks with a clean
`text` (stored, and used to verify quotes) and an `embed_text` (what the
embedding model sees, optionally prefixed with the heading path).

- fixed:     the page as one string (headings inline), cut every N chars with overlap.
- unit:      one chunk per unit: a paragraph, a bullet or a table row.
- structure: adjacent units of the same section merged up to a size limit;
             never crosses a section boundary.
"""

import hashlib
from collections import Counter
from dataclasses import dataclass, field

from langchain_core.documents import Document

FIXED_SIZE = 400
FIXED_OVERLAP = 60  # 15%
STRUCTURE_MAX = 600

# Page-level fields copied from the loader/manifest onto every chunk.
PAGE_FIELDS = ("doc_id", "url", "title", "topic", "updated", "revision_id", "format")


@dataclass
class Chunk:
    text: str
    embed_text: str
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.metadata["chunk_id"]


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _units(section: Document) -> list[str]:
    return [line for line in section.page_content.split("\n") if line.strip()]


def _kind(units: list[str]) -> str:
    kinds = {
        "bullet" if u.startswith("• ") else "table_row" if " | " in u else "paragraph"
        for u in units
    }
    return kinds.pop() if len(kinds) == 1 else "mixed"


def _heading_parts(section: Document) -> dict:
    parts = section.metadata["section_path"].split(" > ")
    return {
        "h2": parts[1] if len(parts) > 1 else "",
        "h3": parts[2] if len(parts) > 2 else "",
    }


def _group_by_page(sections: list[Document]) -> dict[str, list[Document]]:
    pages: dict[str, list[Document]] = {}
    for s in sections:
        pages.setdefault(s.metadata["doc_id"], []).append(s)
    return pages


def _fixed(page: list[Document], size: int, overlap: int) -> list[tuple[Document, str]]:
    """A naive pipeline's view: headings are just lines in the text."""
    lines, offsets, pos = [], [], 0
    for s in page:
        offsets.append((pos, s))
        heading = s.metadata["section_path"].split(" > ")[-1]
        section_lines = ([heading] if heading != s.metadata["title"] else []) + _units(
            s
        )
        lines.extend(section_lines)
        pos += sum(len(line) + 1 for line in section_lines)
    text = "\n".join(lines)
    step = size - overlap
    starts = range(0, max(len(text) - overlap, 1), step)
    # Fixed chunks span sections, so section fields describe where the chunk starts.
    return [
        (
            next(s for off, s in reversed(offsets) if off <= start),
            text[start : start + size],
        )
        for start in starts
    ]


def _structure(section: Document, max_chars: int) -> list[list[str]]:
    groups: list[list[str]] = []
    for unit in _units(section):
        if groups and sum(len(u) + 1 for u in groups[-1]) + len(unit) <= max_chars:
            groups[-1].append(unit)
        else:
            groups.append([unit])
    return groups


def chunk(
    sections: list[Document],
    strategy: str,
    *,
    with_headings: bool,
    fixed_size: int = FIXED_SIZE,
    fixed_overlap: int = FIXED_OVERLAP,
    structure_max: int = STRUCTURE_MAX,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc_id, page in _group_by_page(sections).items():
        doc_hash = _sha1("\n".join(s.page_content for s in page))
        if strategy == "fixed":
            pieces = [
                (s, [text]) for s, text in _fixed(page, fixed_size, fixed_overlap)
            ]
        elif strategy == "unit":
            pieces = [(s, [u]) for s in page for u in _units(s)]
        elif strategy == "structure":
            pieces = [(s, g) for s in page for g in _structure(s, structure_max)]
        else:
            raise ValueError(f"unknown strategy: {strategy}")

        seen: Counter[str] = Counter()
        for index, (section, units) in enumerate(pieces):
            text = "\n".join(units)
            path = section.metadata["section_path"]
            key = f"{doc_id}|{path}|{text}"
            seen[key] += 1  # identical text twice in a section must not collide
            meta = {
                **{
                    k: section.metadata[k]
                    for k in PAGE_FIELDS
                    if section.metadata.get(k) is not None
                },
                **_heading_parts(section),
                "section_path": path,
                "chunk_id": _sha1(f"{key}|{seen[key]}"),
                "doc_hash": doc_hash,
                "chunk_index": index,
                "kind": "text" if strategy == "fixed" else _kind(units),
                "strategy": strategy,
                "with_headings": with_headings,
            }
            if section.metadata.get("page") is not None:
                meta["page"] = section.metadata["page"]
            embed_text = f"{path}\n{text}" if with_headings else text
            chunks.append(Chunk(text=text, embed_text=embed_text, metadata=meta))
    return chunks
