"""PDF loader checks against a real Kol Zchut page.

The fixture pair is the same page saved twice from Chrome: as PDF (what the
loader reads) and as HTML (the answer key, whose text is in correct order).
Fixtures are git-ignored because the content license is unclear, so these
tests skip when the files are missing (e.g. in CI).
"""

import pytest
from answer_key import FIXTURES, docs_text, html_sentences

from rights_rag.loaders.pdf import _fix_ltr_runs, load_pdf

PDF = FIXTURES / "dismissal_guide.pdf"
HTML = FIXTURES / "dismissal_guide.html"

# (fixture name, minimum sentence recall). notice_period scored 94.2% before
# any tuning on it; the remaining misses are math inside Hebrew text, which
# cannot be put back in order from the printed layout alone.
RECALL_CASES = [("dismissal_guide", 0.97), ("notice_period", 0.95)]

needs_fixtures = pytest.mark.skipif(
    not (PDF.exists() and HTML.exists()), reason="fixtures missing"
)


@pytest.mark.parametrize(("name", "min_recall"), RECALL_CASES)
def test_sentence_recall_against_html(name, min_recall):
    pdf, html = FIXTURES / f"{name}.pdf", FIXTURES / f"{name}.html"
    if not (pdf.exists() and html.exists()):
        pytest.skip("fixtures missing")
    sentences = html_sentences(html)
    text = docs_text(load_pdf(pdf))
    missing = [s for s in sentences if s not in text]
    recall = 1 - len(missing) / len(sentences)
    assert recall >= min_recall, f"recall {recall:.1%}, missing e.g.: {missing[:5]}"


@needs_fixtures
def test_sections_and_metadata():
    docs = load_pdf(PDF)
    paths = [d.metadata["section_path"] for d in docs]
    assert "מדריך בנושא פיטורים > תהליך הפיטורים > שימוע" in paths
    assert all(d.metadata["title"] == "מדריך בנושא פיטורים" for d in docs)
    assert all(d.metadata["format"] == "pdf" for d in docs)


@needs_fixtures
def test_table_rows_are_labelled():
    text = "\n".join(d.page_content for d in load_pdf(PDF))
    assert "ותק: יותר מ-3 שנים | ימים להודעה מוקדמת: חודש" in text
    assert "שאלה: המעסיק שלי פיטר אותי בעל פה. האם זה תקין? | תשובה: לא." in text


@pytest.mark.parametrize(
    ("visual", "logical"),
    [
        # Input is what sorting chars right-to-left yields: Hebrew already in
        # reading order, LTR runs reversed.
        ("עד %07 מהשכר", "עד 70% מהשכר"),
        ("טלפון 5656967-470", "טלפון 074-7696565"),
        ("ראו moc.elpmaxe כאן", "ראו example.com כאן"),
        ("סעיף 4א(א) לחוק", "סעיף 4א(א) לחוק"),
    ],
)
def test_fix_ltr_runs(visual, logical):
    assert _fix_ltr_runs(visual) == logical


def test_boilerplate_removed_but_legal_sources_kept():
    pdf = FIXTURES / "notice_period.pdf"
    if not pdf.exists():
        pytest.skip("fixtures missing")
    docs = load_pdf(pdf)
    paths = [d.metadata["section_path"] for d in docs]
    text = "\n".join(d.page_content for d in docs)
    assert not any("גורמים מסייעים" in p for p in paths)
    assert "ראו גם" not in text.splitlines()
    assert "אוחזר מתוך" not in text
    # Ruling summaries are rules a user may ask about, so they must survive.
    assert "ספירת ימי ההודעה המוקדמת היא לפי ימים קלנדריים" in text
