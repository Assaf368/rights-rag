"""Golden set checks. The span check needs the corpus files (git-ignored), so it
skips without them; the helper tests do not."""

import pytest

from rights_rag.golden import (
    GOLDEN,
    assign_splits,
    fact_found,
    lexical_overlap,
    load_golden,
    normalize,
    validate,
)


def test_normalize_drops_direction_marks():
    assert normalize("7.5 ימים -" + chr(0x200F) + "  3") == "7.5 ימים - 3"


def test_fact_found_accepts_any_wording():
    assert fact_found(["לא קיים", "ימים קלנדריים"], normalize("לפי ימים קלנדריים ולא"))


def test_overlap_ignores_hebrew_prefixes():
    assert lexical_overlap("והפיצויים", [["פיצויים"]]) == 1.0


def test_splits_are_stratified_and_kept():
    qs = [{"type": "a"} for _ in range(6)] + [{"type": "b", "split": "dev"}]
    assign_splits(qs)
    assert [q["split"] for q in qs[:6]] == ["dev", "dev", "test"] * 2
    assert qs[-1]["split"] == "dev"


def test_golden_file_shape():
    qs = load_golden()
    assert len({q["id"] for q in qs}) == len(qs)
    for q in qs:
        assert q["split"] in {"dev", "test", "assaf"}
        assert (q["type"] == "unanswerable") == (q["evidence"] == [])


def test_every_span_is_in_the_corpus():
    from rights_rag.corpus import MANIFEST, load_corpus

    if not MANIFEST.exists() or not GOLDEN.exists():
        pytest.skip("corpus missing")
    try:
        sections = load_corpus()
    except (FileNotFoundError, RuntimeError):
        pytest.skip("corpus files missing")
    assert validate(load_golden(), sections) == []
