"""Chunking and incremental indexing, with a fake embedder (no model, no network)."""

import hashlib

import chromadb
import pytest
from langchain_core.documents import Document

from rights_rag.chunking import chunk
from rights_rag.index import sync


def section(path: str, *units: str, doc_id: str = "01") -> Document:
    return Document(
        page_content="\n".join(units),
        metadata={
            "doc_id": doc_id,
            "url": f"https://example.org/{doc_id}",
            "title": path.split(" > ")[0],
            "topic": "t",
            "format": "pdf",
            "section_path": path,
            "page": 1,
            "updated": None,
        },
    )


PAGE = [
    section("דף", "פתיח קצר."),
    section("דף > א", "• אחד " + "x" * 300, "• שתיים " + "y" * 300, "• שלוש."),
    section("דף > ב", "ותק: עד שנה | ימים: חודש", "ותק: שנתיים | ימים: חודש"),
]


def fake_embed(texts):
    return [[b / 255 for b in hashlib.sha1(t.encode()).digest()[:8]] for t in texts]


def test_unit_strategy_one_chunk_per_unit():
    chunks = chunk(PAGE, "unit", with_headings=False)
    assert [c.text for c in chunks] == [
        u for s in PAGE for u in s.page_content.split("\n")
    ]
    assert chunks[-1].metadata["kind"] == "table_row"


def test_structure_merges_within_section_only():
    chunks = chunk(PAGE, "structure", with_headings=True, structure_max=600)
    for c in chunks:
        assert len({c.metadata["section_path"]}) == 1
        assert len(c.text) <= 600 or "\n" not in c.text
    paths = [c.metadata["section_path"] for c in chunks]
    # Section א holds ~620 chars, so it splits in two; section ב fits in one chunk.
    assert paths == ["דף", "דף > א", "דף > א", "דף > ב"]


def test_headings_only_in_embed_text():
    c = chunk(PAGE, "structure", with_headings=True)[1]
    assert c.embed_text.startswith("דף > א\n")
    assert not c.text.startswith("דף")


def test_fixed_overlap():
    long_page = [section("דף", "א" * 1000)]
    chunks = chunk(
        long_page, "fixed", with_headings=False, fixed_size=400, fixed_overlap=60
    )
    assert all(len(c.text) <= 400 for c in chunks)
    assert chunks[0].text[-60:] == chunks[1].text[:60]


def test_ids_are_stable_and_unique():
    a = chunk(PAGE, "unit", with_headings=False)
    b = chunk(PAGE, "unit", with_headings=False)
    assert [c.id for c in a] == [c.id for c in b]
    dup = [section("דף", "לא.", "לא.")]
    assert len({c.id for c in chunk(dup, "unit", with_headings=False)}) == 2


def test_no_none_in_metadata():
    for c in chunk(PAGE, "structure", with_headings=True):
        assert None not in c.metadata.values()  # Chroma rejects None


@pytest.fixture
def collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    return client.get_or_create_collection("test", metadata={"hnsw:space": "cosine"})


def test_sync_is_incremental(collection):
    first = sync(collection, chunk(PAGE, "unit", with_headings=False), fake_embed)
    assert first == {"added": 6, "deleted": 0, "kept": 0}

    again = sync(collection, chunk(PAGE, "unit", with_headings=False), fake_embed)
    assert again == {"added": 0, "deleted": 0, "kept": 6}

    edited = [
        *PAGE[:2],
        section("דף > ב", "ותק: עד שנה | ימים: חודש", "ותק: שנתיים | ימים: 21 יום"),
    ]
    changed = sync(collection, chunk(edited, "unit", with_headings=False), fake_embed)
    assert changed == {"added": 1, "deleted": 1, "kept": 5}
    assert collection.count() == 6


def test_sync_leaves_other_pages_alone(collection):
    sync(collection, chunk(PAGE, "unit", with_headings=False), fake_embed)
    other = [section("אחר", "טקסט.", doc_id="02")]
    sync(collection, chunk(other, "unit", with_headings=False), fake_embed)
    assert collection.count() == 7
