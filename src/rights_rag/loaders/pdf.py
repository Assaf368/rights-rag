"""Load Kol Zchut pages printed to PDF from Chrome.

Chrome writes Hebrew glyphs with correct x positions but in an order that
text extractors read back as reversed words. We rebuild each line by sorting
characters right-to-left by x, then flip runs of left-to-right text (Latin,
digits) back. Headings are detected by font size, which also lets us drop
the print header/footer, link URLs and the language switcher (all small).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

from rights_rag.loaders.common import (
    SENTENCE_END,
    Section,
    clean_title,
    format_rows,
    to_documents,
)

H2_MIN_SIZE = 17.0
H3_MIN_SIZE = 14.5
MIN_TEXT_SIZE = 11.7  # below this: header/footer, link URLs, language switcher
SAME_LINE_TOLERANCE = 2.0
LINK_MERGE_GAP = 12.0  # a word space is ~4pt, table columns are ~36pt apart
PAGE_TOP_MARGIN = (
    50.0  # a header-less table starting this high continues the last page's table
)
PAGE_BOTTOM_MARGIN = 60.0
TABLE_TEXT_SIZE = 12.0  # table rows are body text, never headings
_URL = re.compile(r"[()]?https?://[^\s()]+[()]?")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.:;?!])")

_HEBREW = re.compile(r"[\u0590-\u05FF\u0600-\u06FF]")
_LTR = re.compile(r"[A-Za-z0-9%\u0400-\u04FF]")  # % sticks to its number ("70%")


@dataclass
class _Line:
    y: float
    size: float
    chars: list[dict] = field(default_factory=list)
    # Horizontal extent, including dropped small text (link URLs) so the hole
    # it leaves does not look like a column gap.
    x0: float = float("inf")
    x1: float = float("-inf")

    @property
    def text(self) -> str:
        ordered = sorted(self.chars, key=lambda c: -c["origin"][0])
        return _fix_ltr_runs("".join(c["c"] for c in ordered)).strip()

    def x_gap(self, other: "_Line") -> float:
        return max(other.x0 - self.x1, self.x0 - other.x1, 0)

    def absorb(self, span_or_line, keep_text: bool = True) -> None:
        chars = (
            span_or_line["chars"]
            if isinstance(span_or_line, dict)
            else span_or_line.chars
        )
        if isinstance(span_or_line, _Line):
            self.x0, self.x1 = (
                min(self.x0, span_or_line.x0),
                max(self.x1, span_or_line.x1),
            )
        else:
            self.x0 = min(self.x0, min(c["bbox"][0] for c in chars))
            self.x1 = max(self.x1, max(c["bbox"][2] for c in chars))
        if keep_text:
            self.chars.extend(chars)
            self.size = max(
                self.size,
                span_or_line["size"]
                if isinstance(span_or_line, dict)
                else span_or_line.size,
            )


def _fix_ltr_runs(text: str) -> str:
    """Reverse each run that starts and ends with an LTR char and holds no RTL char."""
    out, i = [], 0
    while i < len(text):
        if not _LTR.match(text[i]):
            out.append(text[i])
            i += 1
            continue
        j, last_ltr = i, i
        while j < len(text) and not _HEBREW.match(text[j]):
            if _LTR.match(text[j]):
                last_ltr = j
            j += 1
        out.append(text[i : last_ltr + 1][::-1])
        i = last_ltr + 1
    return "".join(out)


def _clean(text: str) -> str:
    return _SPACE_BEFORE_PUNCT.sub(r"\1", _URL.sub("", text)).strip()


def _inside(span: dict, rects: list[pymupdf.Rect]) -> bool:
    c = span["chars"][0]["bbox"]
    point = pymupdf.Point((c[0] + c[2]) / 2, (c[1] + c[3]) / 2)
    return any(point in r for r in rects)


@dataclass
class _Table:
    bbox: pymupdf.Rect
    rows: list[list[pymupdf.Rect]]  # cells right-to-left
    has_header: bool


def _tables(page: pymupdf.Page) -> list[_Table]:
    """Tables as Chrome draws them: a white rect holding white cell rects, with a
    coloured rule under the header row. Note boxes hold no white rects, so they
    are not tables. (find_tables() misses most of these: there are no real lines.)
    """
    drawings = page.get_drawings()
    white = [
        d["rect"]
        for d in drawings
        if d.get("fill") == (1.0, 1.0, 1.0) and d["rect"].height > 5
    ]
    rules = [
        d["rect"]
        for d in drawings
        if d.get("fill")
        and d["rect"].height <= 3
        and len(set(d["fill"])) > 1  # coloured, not grey
    ]
    tables = []
    for outer in white:
        cells = [r for r in white if r != outer and r in outer]
        if len(cells) < 2 or any(outer in t.bbox for t in tables):
            continue
        rows: list[list[pymupdf.Rect]] = []
        for cell in sorted(cells, key=lambda r: (round(r.y0), -r.x0)):
            if rows and abs(rows[-1][0].y0 - cell.y0) < 2:
                rows[-1].append(cell)
            else:
                rows.append([cell])
        if max(len(r) for r in rows) < 2:
            continue  # a stack of single cells is a note box, not a table
        first_bottom = max(c.y1 for c in rows[0])
        has_header = any(abs(r.y0 - first_bottom) < 3 and r.width > 50 for r in rules)
        tables.append(_Table(outer, rows, has_header))
    return sorted(tables, key=lambda t: t.bbox.y0)


def _cell_text(spans: list[dict], cell: pymupdf.Rect) -> str:
    lines: list[_Line] = []
    for span in spans:
        if not _inside(span, [cell]):
            continue
        y = span["chars"][0]["origin"][1]
        host = next((l for l in lines if abs(l.y - y) < SAME_LINE_TOLERANCE), None)
        if host is None:
            host = _Line(y=y, size=span["size"])
            lines.append(host)
        host.chars.extend(span["chars"])
    texts = [_clean(l.text) for l in sorted(lines, key=lambda l: l.y)]
    return re.sub(r"•\s+", "• ", " ".join(t for t in texts if t))


def _table_rows(page: pymupdf.Page, table: _Table) -> list[list[str]]:
    spans = [
        s
        for b in page.get_text("rawdict")["blocks"]
        for l in b.get("lines", [])
        for s in l["spans"]
        if s["size"] >= MIN_TEXT_SIZE and s["chars"]
    ]
    return [[_cell_text(spans, c) for c in row] for row in table.rows]


def _page_lines(page: pymupdf.Page, skip: list[pymupdf.Rect]) -> list[_Line]:
    """Visual lines in content-stream order, skipping small text and tables.

    Chrome emits link text as a separate block after the page body, so a span
    joins an earlier line when it sits on the same baseline right beside it.
    Table columns on the same baseline are further apart and stay separate.
    """
    lines: list[_Line] = []
    for block in page.get_text("rawdict")["blocks"]:
        for raw_line in block.get("lines", []):
            for segment in _segments(raw_line, skip):
                host = next(
                    (
                        l
                        for l in lines
                        if abs(l.y - segment.y) < SAME_LINE_TOLERANCE
                        and l.x_gap(segment) < LINK_MERGE_GAP
                    ),
                    None,
                )
                if host is None:
                    lines.append(segment)
                else:
                    host.absorb(segment)
    return lines


def _segments(raw_line: dict, skip: list[pymupdf.Rect]) -> list[_Line]:
    """Split one extractor line where there is a column-sized horizontal gap.
    Spans are not stored in x order, so sort them first."""
    spans = [s for s in raw_line["spans"] if s["chars"] and not _inside(s, skip)]
    spans.sort(key=lambda s: min(c["bbox"][0] for c in s["chars"]))
    segments: list[_Line] = []
    for span in spans:
        x0 = min(c["bbox"][0] for c in span["chars"])
        if not segments or x0 - segments[-1].x1 > LINK_MERGE_GAP:
            segments.append(_Line(y=span["chars"][0]["origin"][1], size=0))
        segments[-1].absorb(span, keep_text=span["size"] >= MIN_TEXT_SIZE)
    for seg in segments:
        if seg.chars:
            seg.y = seg.chars[0]["origin"][1]
    return [seg for seg in segments if seg.chars]


def _paragraphs(doc: pymupdf.Document):
    """Yield (page_number, size, text, is_table_row) paragraphs in reading order."""
    pending_bullet = False
    carried_header = None  # header of a table that may continue on the next page

    def table_paragraphs(page: pymupdf.Page, table: _Table) -> list[str]:
        nonlocal carried_header
        rows = _table_rows(page, table)
        if table.has_header:
            header, rows = rows[0], rows[1:]
        elif table.bbox.y0 < PAGE_TOP_MARGIN:
            header = carried_header
        else:
            header = None
        carried_header = (
            header if table.bbox.y1 > page.rect.height - PAGE_BOTTOM_MARGIN else None
        )
        return format_rows(header, rows)

    for page in doc:
        # Note boxes ("לדוגמה", "שימו לב") are drawn like tables but have no
        # header; they read fine as ordinary paragraphs.
        tables = [
            t
            for t in _tables(page)
            if t.has_header or (carried_header and t.bbox.y0 < PAGE_TOP_MARGIN)
        ]
        para, prev = [], None
        # Top-to-bottom order, not content-stream order: Chrome writes some text
        # late (link text, FAQ accordion questions after all the answers).
        lines = sorted(_page_lines(page, [t.bbox for t in tables]), key=lambda l: l.y)
        for line in lines:
            text = _clean(line.text)
            if not text:
                continue
            # Emit a table where it sits in the flow: before the first line below its top.
            while tables and tables[0].bbox.y0 < line.y:
                if para:
                    yield page.number + 1, prev.size, " ".join(para), False
                    para, prev = [], None
                for row in table_paragraphs(page, tables.pop(0)):
                    yield page.number + 1, TABLE_TEXT_SIZE, row, True
            if text == "•":
                pending_bullet = True
                continue
            starts_bullet = pending_bullet or text.startswith("•")
            new_para = prev is not None and (
                starts_bullet
                or abs(line.size - prev.size) > 0.1
                or line.y - prev.y > 1.8 * line.size
                or line.y < prev.y  # jumped back up: next table column
            )
            if para and new_para:
                yield page.number + 1, prev.size, " ".join(para), False
                para = []
            if pending_bullet:
                text = "• " + text
                pending_bullet = False
            para.append(text)
            prev = line
        if para:
            yield page.number + 1, prev.size, " ".join(para), False
        for table in tables:
            for row in table_paragraphs(page, table):
                yield page.number + 1, TABLE_TEXT_SIZE, row, True


def load_pdf(path: str | Path) -> list[Document]:
    """Return one Document per section, with the heading path in metadata."""
    path = Path(path)
    with pymupdf.open(path) as doc:
        title = clean_title(doc.metadata.get("title") or path.stem)
        paragraphs = list(_paragraphs(doc))

    sections: list[Section] = []
    h2 = h3 = None
    last_page, last_is_table = 0, False
    for page, size, text, is_table_row in paragraphs:
        if size >= H2_MIN_SIZE and text == title and not sections:
            continue  # the page title itself
        if size >= H3_MIN_SIZE:
            if size >= H2_MIN_SIZE:
                h2, h3 = text, None
            else:
                h3 = text
            sections.append(Section(h2, h3, page))
            continue
        if not sections:
            sections.append(Section(None, None, page))
        s = sections[-1]
        # A paragraph cut by a page break continues on the next page.
        if (
            s.texts
            and last_page < page
            and not last_is_table
            and not is_table_row
            and not SENTENCE_END.search(s.texts[-1])
            and not text.startswith("•")
        ):
            s.texts[-1] += " " + text
        else:
            s.texts.append(text)
        last_page, last_is_table = page, is_table_row

    return to_documents(sections, title=title, source=str(path), fmt="pdf")
