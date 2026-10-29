"""Tokenizer, RRF and BM25 retrieval with a fake collection (no model)."""

from rights_rag.retrieval import Retriever, rrf, tokenize


def test_tokenize_keeps_numbers_and_acronyms_whole():
    assert tokenize('22.60 ₪, 6,443.85, 70%, חל"ת, טופס 161') == [
        "22.60",
        "6,443.85",
        "70%",
        'חל"ת',
        "טופס",
        "161",
    ]


def test_prefix_stripping_adds_stems_and_keeps_original():
    assert tokenize("והפיצויים", strip_prefixes=True) == [
        "והפיצויים",
        "הפיצויים",
        "פיצויים",
    ]
    # Short words are left alone: "שלי" is a word, not "ש" + "לי".
    assert tokenize("שלי", strip_prefixes=True) == ["שלי"]


def test_rrf_rewards_agreement():
    # "c" is 3rd and 1st; "a" is 1st and 2nd. Both beat "b" and "d", which appear once.
    assert rrf([["a", "b", "c"], ["c", "a", "d"]]) == ["a", "c", "b", "d"]


class FakeCollection:
    def __init__(self, docs):
        self.docs = docs

    def get(self, include):
        ids = list(self.docs)
        return {
            "ids": ids,
            "documents": list(self.docs.values()),
            "metadatas": [
                {"embed_text": t, "doc_id": "01"} for t in self.docs.values()
            ],
        }

    def query(self, query_embeddings, n_results, include):
        return {"ids": [list(self.docs)[:n_results]]}


def test_bm25_needs_prefix_stripping_for_hebrew():
    col = FakeCollection(
        {
            "x": "לעובד מגיעים והפיצויים בסיום",
            "y": "דמי הבראה לפי ותק",
            "z": "חופשה שנתית",
        }
    )
    r = Retriever(col, query_embedder=lambda q: [0.0])
    assert "x" not in r.bm25("פיצויים")
    assert r.bm25("פיצויים", strip_prefixes=True)[0] == "x"


def test_rewrite_parses_one_query_per_line():
    from rights_rag.answer import Answer
    from rights_rag.rewrite import rewrite

    def fake(system, user):
        return Answer(
            "1. פיטורים ללא הודעה מוקדמת\n\n- זכויות בפיטורים\nא\nב",
            "end_turn",
            0,
            0,
            "x",
        )

    assert rewrite("עפו עליי", fake) == [
        "פיטורים ללא הודעה מוקדמת",
        "זכויות בפיטורים",
        "א",
    ]
