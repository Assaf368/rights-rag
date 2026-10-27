"""Answer key for loader tests: the sentences of a Kol Zchut page, taken from
its HTML (text in correct order), minus what the loaders drop on purpose."""

import re
from pathlib import Path

from bs4 import BeautifulSoup

from rights_rag.loaders.common import NOISE_SECTIONS

FIXTURES = Path(__file__).parent / "fixtures"


def norm(s: str) -> str:
    s = s.replace("\u200f", "").replace("\u200e", "").replace("־", "-")
    s = re.sub(r"\s+([,.:;?!])", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def html_sentences(path: Path) -> set[str]:
    """Sentences of the HTML page, minus what the loaders drop on purpose."""
    body = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser").select_one(
        ".mw-parser-output"
    )
    # .noprint is hidden when printing, so it never reaches the PDF.
    for junk in body.select("style, script, .noprint, .toc, #toc, .mw-editsection"):
        junk.decompose()
    drop_section(body, NOISE_SECTIONS)
    # "ראו גם" lists and "לרשימת ..." pointers: list items that are just one link.
    for li in body.find_all("li"):
        links = li.find_all("a")
        if len(links) == 1 and norm(links[0].get_text()) == norm(
            li.get_text().lstrip("לרשימת ")
        ):
            li.decompose()
    blocks = [
        norm(el.get_text()) for el in body.find_all(["p", "li", "td", "h2", "h3", "h4"])
    ]
    return {
        s.strip()
        for b in blocks
        for s in re.split(r"(?<=[.?!:])\s+", b)
        if len(s.strip()) >= 25 and not s.startswith(("לרשימת", "לערך מורחב"))
    }


def drop_section(body, headings: set[str]) -> None:
    for h2 in body.find_all("h2"):
        if norm(h2.get_text()) not in headings:
            continue
        for sibling in list(h2.find_all_next()):
            if sibling.name == "h2":
                break
            sibling.decompose()
        h2.decompose()


def docs_text(docs) -> str:
    """All loaded text as one normalized string, to search sentences in."""
    # Headings live in metadata, not in page_content, so add them back.
    return norm(
        " ".join(
            d.metadata["section_path"].split(" > ")[-1]
            + " "
            + d.page_content.replace("\n", " ").replace("• ", "")
            for d in docs
        )
    )
