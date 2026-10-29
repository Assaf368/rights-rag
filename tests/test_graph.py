"""The v3 graph with fake models: which steps run for each critic decision."""

from rights_rag.answer import Answer
from rights_rag.graph import build_graph, run
from rights_rag.prompts import ABSTAIN_V1, UNVERIFIED_V2

GOOD = "מגיע לך «באורך מצטבר של 45 דקות» [1]."
CHUNKS = [{"text": "עובד זכאי להפסקות באורך מצטבר של 45 דקות."}]


def make(answers: list[str], action: str | None):
    log = {"search": [], "answer": 0, "critic": 0, "rewrite": 0}

    def search(queries):
        log["search"].append(queries)
        return CHUNKS

    def answer(question, chunks):
        log["answer"] += 1
        return Answer(answers[log["answer"] - 1], "end_turn", 10, 5, "fake")

    def critique(question, chunks, text, reason):
        log["critic"] += 1
        return {"action": action, "reason": "x"}, None

    def rewrite(question):
        log["rewrite"] += 1
        return ["הפסקות בעבודה"], None

    return build_graph(search, answer, critique, rewrite), log


def test_unflagged_answer_skips_the_critic():
    graph, log = make([GOOD], "retry_with_rewrite")
    out = run(graph, "כמה הפסקה?")
    assert out["answer"]["text"] == GOOD and log["critic"] == 0
    assert log["search"] == [["כמה הפסקה?"]]


def test_retry_rewrites_searches_again_and_answers_once_more():
    graph, log = make([ABSTAIN_V1, GOOD], "retry_with_rewrite")
    out = run(graph, "כמה הפסקה?", sources=CHUNKS)
    assert out["answer"]["text"] == GOOD and out["rounds"] == 1
    assert log["search"] == [
        ["כמה הפסקה?", "הפסקות בעבודה"]
    ]  # first round reused sources
    assert log["critic"] == 1 and log["answer"] == 2  # no second review round


def test_critic_abstain_and_accept():
    graph, _ = make([UNVERIFIED_V2], "abstain")  # blocked by the gate, so flagged
    assert run(graph, "כמה הפסקה?")["answer"]["text"] == ABSTAIN_V1
    graph, log = make([ABSTAIN_V1], "accept")
    assert (
        run(graph, "כמה הפסקה?")["answer"]["text"] == ABSTAIN_V1 and log["answer"] == 1
    )
