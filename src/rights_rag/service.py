"""One question in, one answer out: what the CLI and the API share."""

import time

from rights_rag.answer import check_citations
from rights_rag.costs import cost
from rights_rag.events import emit, listening, public_sources
from rights_rag.golden import normalize
from rights_rag.graph import build_graph, run
from rights_rag.prompts import ABSTAIN_V1, DISCLAIMER_V1, UNVERIFIED_V2


def to_response(state: dict, seconds: float) -> dict:
    """The graph's final state as a plain response: the answer, numbered
    sources, each citation checked against its source, and how it got there."""
    text = state["answer"]["text"]
    sources = state["sources"]
    return {
        "answer": text,
        "status": (
            "blocked"
            if text == UNVERIFIED_V2
            else "abstained"
            if normalize(ABSTAIN_V1) in normalize(text)
            else "answered"
        ),
        "disclaimer": DISCLAIMER_V1,
        "sources": public_sources(sources),
        "citations": check_citations(text, sources),
        "cost_usd": round(sum(cost(c) for c in state.get("calls") or []), 5),
        "seconds": round(seconds, 2),
    }


class Answerer:
    """Holds the loaded models and answers with v2.1. The review round (v3)
    was measured and left out: on held-out data it rescued no answers and
    cost four times as much (see the README). It remains in graph.py for the
    evaluation."""

    def __init__(self, components: dict):
        self.graph = build_graph(**components | {"critique": None})

    def ask(self, question: str) -> dict:
        start = time.perf_counter()
        state = run(self.graph, question)
        return to_response(state, time.perf_counter() - start)

    def ask_streaming(self, question: str, on_event) -> None:
        """ask(), reporting progress to on_event as it happens. The last
        event is "done" with the full response, or "error"."""
        with listening(on_event):
            try:
                emit("done", response=self.ask(question))
            except Exception as e:  # noqa: BLE001 - the stream must end with an event
                emit("error", message=f"{type(e).__name__}: {e}")


def default_answerer() -> Answerer:
    from dotenv import load_dotenv

    from rights_rag.graph import default_components

    load_dotenv()
    return Answerer(default_components())
