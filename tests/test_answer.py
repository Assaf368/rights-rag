"""Prompts and answer scoring with a fake model (no API calls)."""

from rights_rag.answer import Answer, answer
from rights_rag.eval_answer import check_citations, is_abstention, keys_found, score
from rights_rag.prompts import ABSTAIN_V1, render_v0, render_v1

CHUNKS = [
    {
        "text": "עובד זכאי להפסקות באורך מצטבר של 45 דקות.",
        "section_path": "הפסקות",
        "url": "u1",
        "updated": None,
    },
    {
        "text": "שכר המינימום הוא 35.40 ₪ לשעה.",
        "section_path": "שכר",
        "url": "u2",
        "updated": "2026-04-01",
    },
]


def test_v0_is_bare_and_v1_numbers_sources():
    system, user = render_v0("כמה הפסקה?", CHUNKS)
    assert system is None and "<source" not in user and "45 דקות" in user
    system, user = render_v1("כמה הפסקה?", CHUNKS)
    assert ABSTAIN_V1 in system
    assert '<source id="2">' in user and "updated: unknown" in user
    assert user.rstrip().endswith("Question: כמה הפסקה?")


def test_answer_passes_rendered_prompt_to_model():
    seen = {}

    def fake(system, user):
        seen.update(system=system, user=user)
        return Answer("45 דקות", "end_turn", 10, 5, "fake")

    assert answer("כמה הפסקה?", CHUNKS, "v1", fake).text == "45 דקות"
    assert seen["system"] and "Question: כמה הפסקה?" in seen["user"]


def test_abstention_detection():
    assert is_abstention(ABSTAIN_V1)
    assert is_abstention("המקורות אינם מתייחסים לנושא הזה.")
    assert not is_abstention("מגיעות לך 45 דקות הפסקה.")


def test_keys_need_every_fact():
    assert keys_found([["45 דקות"], ["הפסקה", "הפסקות"]], "מגיעות 45 דקות של הפסקות")
    assert not keys_found([["45 דקות"], ["150%"]], "מגיעות 45 דקות")


def test_citations_checked_against_their_own_source():
    text = "מגיע לך «באורך מצטבר של 45 דקות» [1], והשכר «35.40 ₪ לשעה» [1]."
    result = check_citations(text, CHUNKS)
    assert [c["valid"] for c in result] == [
        True,
        False,
    ]  # second quote is in source 2, not 1


def test_score_end_to_end():
    questions = [
        {"id": "a", "evidence": [["45 דקות"]], "answer_keys": [["45 דקות"]]},
        {"id": "u", "evidence": []},
    ]
    run = {
        "sources": {"a": CHUNKS, "u": CHUNKS},
        "answers": {
            "v1": {
                "a": {
                    "text": "«45 דקות» [1]",
                    "stop_reason": "end_turn",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "model": "x",
                },
                "u": {
                    "text": ABSTAIN_V1,
                    "stop_reason": "end_turn",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "model": "x",
                },
            }
        },
    }
    r = score(questions, run)["v1"]
    assert r["key_facts_correct"] == 1.0
    assert r["abstained_when_unanswerable"] == 1.0
    assert r["abstained_when_answerable"] == 0.0
    assert r["citations_valid"] == 1.0


def test_judge_label_parsing_does_not_read_incorrect_as_correct():
    from rights_rag.eval_answer import _LABEL

    assert _LABEL.search("INCORRECT").group(1) == "INCORRECT"
    assert _LABEL.search("LABEL: CORRECT.").group(1) == "CORRECT"
    assert _LABEL.search("UNSUPPORTED - A NUMBER FROM MEMORY").group(1) == "UNSUPPORTED"
