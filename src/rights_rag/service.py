"""One question in, one answer out: what the CLI and the API share."""

import time

from rights_rag.answer import check_citations
from rights_rag.critic import flag
from rights_rag.eval_answer import _cost
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
        "sources": [
            {
                "n": n,
                "title": c["title"],
                "section": c["section_path"],
                "url": c["url"],
                "updated": c.get("updated") or None,
                "text": c["text"],
            }
            for n, c in enumerate(sources, start=1)
        ],
        "citations": check_citations(text, sources),
        "review": {
            "flag": flag(state["answer"]),  # of the final answer
            "decision": state.get("decision"),  # with the flag that triggered it
            "searched_again": len(state.get("queries") or []) > 1,
            "queries": state.get("queries") or [],
        },
        "cost_usd": round(sum(_cost(c) for c in state.get("calls") or []), 5),
        "seconds": round(seconds, 2),
    }


class Answerer:
    """Holds the loaded models; ask() runs v3, or v2.1 with critic=False."""

    def __init__(self, components: dict):
        self.graphs = {
            True: build_graph(**components),
            False: build_graph(**components | {"critique": None}),
        }

    def ask(self, question: str, critic: bool = True) -> dict:
        start = time.perf_counter()
        state = run(self.graphs[critic], question)
        return to_response(state, time.perf_counter() - start)


def default_answerer() -> Answerer:
    from dotenv import load_dotenv

    from rights_rag.graph import default_components

    load_dotenv()
    return Answerer(default_components())
