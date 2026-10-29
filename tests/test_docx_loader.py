"""DOCX loader checks. Fixtures are HTML pages converted to Word on the fly,
so the HTML is both the source and the answer key."""

import pytest
from answer_key import FIXTURES, docs_text, html_sentences

from rights_rag.html_to_docx import convert
from rights_rag.loaders.docx import load_docx
from rights_rag.loaders.pdf import load_pdf

NAMES = ["dismissal_guide", "notice_period"]
MIN_RECALL = 0.99  # no layout guessing here, so nearly everything must survive


@pytest.fixture
def as_docx(tmp_path):
    def _convert(name):
        html = FIXTURES / f"{name}.html"
        if not html.exists():
            pytest.skip("fixtures missing")
        out = tmp_path / f"{name}.docx"
        convert(html, out)
        return out

    return _convert


@pytest.mark.parametrize("name", NAMES)
def test_sentence_recall_against_html(name, as_docx):
    text = docs_text(load_docx(as_docx(name)))  # skips first if fixtures are missing
    sentences = html_sentences(FIXTURES / f"{name}.html")
    missing = [s for s in sentences if s not in text]
    recall = 1 - len(missing) / len(sentences)
    assert recall >= MIN_RECALL, f"recall {recall:.1%}, missing e.g.: {missing[:5]}"


@pytest.mark.parametrize("name", NAMES)
def test_same_structure_as_pdf_loader(name, as_docx):
    """The same page must come out the same whichever format it was stored in."""
    pdf = FIXTURES / f"{name}.pdf"
    if not pdf.exists():
        pytest.skip("fixtures missing")
    from_pdf, from_docx = load_pdf(pdf), load_docx(as_docx(name))
    assert [d.metadata["section_path"] for d in from_docx] == [
        d.metadata["section_path"] for d in from_pdf
    ]
    assert from_docx[0].metadata.keys() == from_pdf[0].metadata.keys()
    assert {d.metadata["format"] for d in from_docx} == {"docx"}


def test_noise_removed(as_docx):
    text = "\n".join(d.page_content for d in load_docx(as_docx("notice_period")))
    assert "בחרו בכל זכות כאתר מועדף בגוגל" not in text
    assert "תוכן עניינים" not in text
    assert "ראו גם" not in text.splitlines()
    assert "ספירת ימי ההודעה המוקדמת היא לפי ימים קלנדריים" in text


def test_table_rows_are_labelled(as_docx):
    text = "\n".join(d.page_content for d in load_docx(as_docx("dismissal_guide")))
    assert "ותק: יותר מ-3 שנים | ימים להודעה מוקדמת: חודש" in text
