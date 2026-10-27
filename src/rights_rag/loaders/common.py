"""Pieces shared by the PDF and DOCX loaders, so both produce the same output:
one Document per section, the same metadata keys, the same table-row format
and the same boilerplate removed."""

import re
from dataclasses import dataclass, field

from langchain_core.documents import Document

TITLE_SUFFIX = " – כל-זכות"

# Kol Zchut boilerplate: contact lists and link lists, no rights content.
# ("מקורות משפטיים ורשמיים" stays: it holds ruling summaries and law sections.)
NOISE_SECTIONS = {"גורמים מסייעים"}
SEE_ALSO = "ראו גם"
PRINT_FOOTER = "אוחזר מתוך"
# Screen-only widgets (hidden when printing, so only HTML-based formats have them).
SCREEN_ONLY = (
    "בחרו בכל זכות כאתר מועדף בגוגל",
    "בלחיצה תועברו למסך של Google",
    "תוכן עניינים",
)

SENTENCE_END = re.compile(r"[.:?!]$")
_WORD = re.compile(r"\w")
_LINK_POINTER = re.compile(r"^• (?:לרשימת|לערך מורחב)")  # "for the list of ... see"
_PAGE_TYPE = re.compile(r" \((?:זכות|מושג|הליך|פסק דין|חוק)\)$")  # page-type label


@dataclass
class Section:
    h2: str | None
    h3: str | None
    page: int | None = None
    texts: list[str] = field(default_factory=list)


def clean_title(raw: str) -> str:
    """Drop the site suffix and page-type label:
    "פיצויי פיטורים לעובד שפוטר (זכות) – כל-זכות" -> "פיצויי פיטורים לעובד שפוטר"."""
    return _PAGE_TYPE.sub("", raw.strip().removesuffix(TITLE_SUFFIX))


def format_rows(header: list[str] | None, rows: list[list[str]]) -> list[str]:
    """One paragraph per row; with a header, each value is labelled ("ותק: עד שנה")."""
    out = []
    for row in rows:
        if not any(row):
            continue
        if header and len(row) == len(header):
            out.append(
                " | ".join(f"{h}: {v}" for h, v in zip(header, row, strict=True) if v)
            )
        else:
            out.append(" | ".join(v for v in row if v))
    return out


def drop_boilerplate(texts: list[str]) -> list[str]:
    """Remove the "ראו גם" link list, bare link pointers, the print footer,
    screen-only widgets and stray punctuation.

    After "ראו גם" only short lines without a sentence end (page titles) are
    dropped, so a real sentence that follows is kept.
    """
    out, in_see_also = [], False
    for text in texts:
        if text == SEE_ALSO:
            in_see_also = True
            continue
        if in_see_also and len(text) < 120 and not SENTENCE_END.search(text):
            continue
        in_see_also = False
        if (
            text.startswith(PRINT_FOOTER)
            or text.startswith(SCREEN_ONLY)
            or _LINK_POINTER.match(text)
            or not _WORD.search(text)
        ):
            continue
        out.append(text)
    return out


def to_documents(
    sections: list[Section], *, title: str, source: str, fmt: str
) -> list[Document]:
    docs = []
    for s in sections:
        if s.h2 in NOISE_SECTIONS:
            continue
        texts = drop_boilerplate(s.texts)
        if not texts:
            continue
        path_parts = [title] + [h for h in (s.h2, s.h3) if h]
        docs.append(
            Document(
                page_content="\n".join(texts),
                metadata={
                    "source": source,
                    "format": fmt,
                    "title": title,
                    "section_path": " > ".join(path_parts),
                    "page": s.page,
                },
            )
        )
    return docs
