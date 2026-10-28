"""Retrieval scoring on hand-made hits (no model, no index)."""

from rights_rag.eval_retrieval import first_full_rank, score


def hit(text, doc="01", index=0, strategy="unit"):
    return {"text": text, "doc_id": doc, "chunk_index": index, "strategy": strategy}


Q1 = {"id": "q1", "type": "number", "evidence": [["מגיעים 14 יום"]]}
Q2 = {
    "id": "q2",
    "type": "multi_fact",
    "evidence": [["עובדה א"], ["עובדה ב", "ניסוח אחר של ב"]],
}
QU = {"id": "qu", "type": "unanswerable", "evidence": []}


def test_rank_of_single_fact():
    hits = [hit("משהו אחר"), hit("לפי החוק מגיעים 14 יום הודעה"), hit("עוד")]
    assert first_full_rank(Q1, hits) == 2


def test_multi_fact_needs_all_facts():
    hits = [hit("עובדה א כאן"), hit("לא קשור"), hit("ניסוח אחר של ב")]
    assert first_full_rank(Q2, hits) == 3
    assert first_full_rank(Q2, hits[:2]) is None


def test_fixed_chunks_joined_across_boundary():
    # A fact longer than the 60-char overlap, cut by the boundary: neither chunk
    # holds all of it, only the two joined do.
    span = "א" * 50 + " אמצע " + "ב" * 50
    q = {"id": "q", "type": "number", "evidence": [[span]]}
    a = "x" * 320 + span[:80]
    b = a[-60:] + span[80:]
    hits = [hit(a, index=3, strategy="fixed"), hit(b, index=4, strategy="fixed")]
    assert first_full_rank(q, hits) == 2
    # Not adjacent: no join.
    hits[1]["chunk_index"] = 7
    assert first_full_rank(q, hits) is None


def test_score_skips_unanswerable_and_computes_mrr():
    run = {
        "col": {
            "q1": [hit("מגיעים 14 יום")],
            "q2": [hit("לא"), hit("עובדה א ועובדה ב")],
            "qu": [hit("כלום")],
        }
    }
    r = score([Q1, Q2, QU], run)["col"]
    assert r["n"] == 2
    assert r["hit@1"] == 0.5
    assert r["hit@3"] == 1.0
    assert r["mrr"] == (1 + 1 / 2) / 2
