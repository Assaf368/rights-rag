"""The service and the REST API with fake models."""

import json

from fastapi.testclient import TestClient

from rights_rag.answer import Answer
from rights_rag.api import create_app
from rights_rag.prompts import ABSTAIN_V1
from rights_rag.service import Answerer

CHUNKS = [
    {
        "text": "עובד זכאי להפסקות באורך מצטבר של 45 דקות.",
        "title": "הפסקות בעבודה",
        "section_path": "הפסקות בעבודה > אורך ההפסקה",
        "url": "https://www.kolzchut.org.il/he/x",
        "updated": "",
    }
]


def fake_answerer(first_text: str = "«באורך מצטבר של 45 דקות» [1]") -> Answerer:
    texts = iter([first_text, "«באורך מצטבר של 45 דקות» [1]"])
    return Answerer(
        {
            "search": lambda queries: CHUNKS,
            "answer": lambda q, chunks: Answer(
                next(texts), "end_turn", 1000, 100, "claude-haiku-5-5"
            ),
            "critique": lambda q, c, t, r: (
                {"action": "retry_with_rewrite", "reason": "x"},
                None,
            ),
            "rewrite": lambda q: (["הפסקות"], None),
        }
    )


def test_response_has_checked_citations_and_cost():
    r = fake_answerer().ask("כמה הפסקה?")
    assert r["status"] == "answered" and r["citations"][0]["valid"]
    assert r["sources"][0]["n"] == 1 and r["sources"][0]["updated"] is None
    assert r["cost_usd"] > 0


def test_the_service_never_runs_the_review_round():
    # The components include a critic that would always search again; the
    # service answers with v2.1, so an abstention stays an abstention.
    r = fake_answerer(ABSTAIN_V1).ask("כמה הפסקה?")
    assert r["status"] == "abstained"


def test_api_ask_and_validation():
    with TestClient(create_app(fake_answerer)) as client:
        assert client.get("/health").json() == {"ok": True}
        r = client.post("/ask", json={"question": "כמה הפסקה?"})
        assert r.status_code == 200 and r.json()["status"] == "answered"
        assert client.post("/ask", json={"question": ""}).status_code == 422


def stream_events(make_answerer) -> list[dict]:
    with (
        TestClient(create_app(make_answerer)) as client,
        client.stream("POST", "/ask/stream", json={"question": "כמה הפסקה?"}) as r,
    ):
        return [
            json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")
        ]


def test_stream_sends_progress_then_the_full_response():
    events = stream_events(lambda: fake_answerer(ABSTAIN_V1))
    types = [e["type"] for e in events]
    assert types[0] == "stage" and types[-1] == "done"
    assert "sources" in types and "token" not in types  # the fake does not stream
    stages = [e["stage"] for e in events if e["type"] == "stage"]
    assert stages == ["searching", "answering"]
    assert events[-1]["response"]["status"] == "abstained"


def test_stream_ends_with_an_error_event():
    def broken():
        a = fake_answerer()
        a.graph = None  # run() will fail
        return a

    assert stream_events(broken)[-1]["type"] == "error"
