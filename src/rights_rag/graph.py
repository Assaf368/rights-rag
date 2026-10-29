"""v3: the answer pipeline as a LangGraph graph, with one review round.

    retrieve -> answer -> review -+- not flagged, or critic accepts --> END
                                  +- critic: abstain -----------------> abstain -> END
                                  +- critic: retry_with_rewrite -> rewrite -> retrieve -> answer -> END

The answer step is v2.1 (prompt v1.2 behind the citation gate, one quote
retry, abstention sentence enforced). Code decides when to review (cheap
flags); the critic decides what to do; the loop runs at most once.
"""

from collections.abc import Callable
from dataclasses import asdict
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from rights_rag.answer import Answer
from rights_rag.prompts import ABSTAIN_V1

ANSWER_VERSION = "v2.1"
MAX_ROUNDS = 1


class State(TypedDict, total=False):
    question: str
    queries: list[str]  # the question first, then any rewrites
    sources: list[dict]
    answer: dict  # a saved Answer
    flag: str | None
    critic: dict | None  # the pending decision, cleared once acted on
    decision: dict | None  # the critic's decision, kept for the record
    rounds: int
    calls: list[dict]  # model, input_tokens, output_tokens of every call


def _usage(a: Answer) -> dict:
    return {
        "model": a.model,
        "input_tokens": a.input_tokens,
        "output_tokens": a.output_tokens,
    }


def build_graph(
    search: Callable[[list[str]], list[dict]],
    answer: Callable[[str, list[dict]], Answer],
    critique: Callable[[str, list[dict], str, str], tuple[dict, Answer | None]],
    rewrite: Callable[[str], tuple[list[str], Answer | None]],
):
    """search: queries -> top chunks; answer: (question, chunks) -> Answer;
    critique: (question, chunks, answer text, flag) -> (verdict, usage);
    rewrite: question -> (queries, usage). Injected so tests need no models."""
    from rights_rag.critic import flag

    def retrieve(state: State) -> State:
        if state.get("sources") is not None and not state.get("rounds"):
            return {}  # first round with sources already retrieved (evaluation)
        return {"sources": search(state["queries"])}

    def respond(state: State) -> State:
        a = answer(state["question"], state["sources"])
        return {"answer": asdict(a), "calls": [*state.get("calls", []), _usage(a)]}

    def review(state: State) -> State:
        reason = flag(state["answer"])
        if reason is None or state.get("rounds", 0) >= MAX_ROUNDS:
            return {"flag": reason}
        verdict, usage = critique(
            state["question"], state["sources"], state["answer"]["text"], reason
        )
        calls = state.get("calls", []) + ([_usage(usage)] if usage else [])
        return {"flag": reason, "critic": verdict, "decision": verdict, "calls": calls}

    def after_review(state: State) -> str:
        if state.get("rounds", 0) >= MAX_ROUNDS or not state.get("critic"):
            return END
        return {"accept": END, "abstain": "abstain"}.get(
            state["critic"]["action"], "rewrite"
        )

    def do_rewrite(state: State) -> State:
        queries, usage = rewrite(state["question"])
        calls = state.get("calls", []) + ([_usage(usage)] if usage else [])
        return {
            "queries": [state["question"], *queries],
            "rounds": state.get("rounds", 0) + 1,
            "critic": None,
            "calls": calls,
        }

    def abstain(state: State) -> State:
        return {"answer": {**state["answer"], "text": ABSTAIN_V1}}

    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", respond)
    g.add_node("review", review)
    g.add_node("rewrite", do_rewrite)
    g.add_node("abstain", abstain)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", "review")
    g.add_conditional_edges("review", after_review, ["rewrite", "abstain", END])
    g.add_edge("rewrite", "retrieve")
    g.add_edge("abstain", END)
    return g.compile()


def run(graph, question: str, sources: list[dict] | None = None) -> State:
    """Answer one question. Pass sources to skip the first retrieval."""
    state: State = {"question": question, "queries": [question], "rounds": 0}
    if sources is not None:
        state["sources"] = sources
    return graph.invoke(state)


def _recording(call):
    """Wrap a model call so the graph can count what it cost."""
    seen: list[Answer] = []

    def wrapped(system, user):
        a = call(system, user)
        seen.append(a)
        return a

    return wrapped, seen


def default_graph(answer_model: str | None = None):
    """The graph with the real retriever, Haiku answers and the Sonnet critic."""
    from rights_rag import critic as critic_mod
    from rights_rag.answer import ANSWER_MODEL, claude_caller
    from rights_rag.answer import answer as answer_fn
    from rights_rag.eval_answer import COLLECTION, RERANK_DEPTH, TOP_K
    from rights_rag.index import E5Embedder, open_collection
    from rights_rag.retrieval import Reranker, Retriever
    from rights_rag.rewrite import REWRITE_MODEL
    from rights_rag.rewrite import rewrite as rewrite_fn

    retriever = Retriever(open_collection(COLLECTION), E5Embedder().query)
    reranker = Reranker()
    answer_call = claude_caller(model=answer_model or ANSWER_MODEL)
    critic_call = claude_caller(model=critic_mod.CRITIC_MODEL)
    rewrite_call = claude_caller(model=REWRITE_MODEL)

    def search(queries: list[str]) -> list[dict]:
        ids = list(
            dict.fromkeys(
                c for q in queries for c in retriever.vector(q, n=RERANK_DEPTH)
            )
        )
        pairs = [(c, retriever.chunks[c]["embed_text"]) for c in ids]
        ranked = reranker.rerank_many(queries, pairs)
        return [
            {k: v for k, v in hit.items() if k != "embed_text"}
            for hit in retriever.hits(ranked, TOP_K)
        ]

    def critique(question, chunks, text, reason):
        call, seen = _recording(critic_call)
        verdict = critic_mod.critique(question, chunks, text, reason, call)
        return verdict, (seen[0] if seen else None)

    def rewrite(question):
        call, seen = _recording(rewrite_call)
        return rewrite_fn(question, call), (seen[0] if seen else None)

    return build_graph(
        search,
        lambda q, chunks: answer_fn(q, chunks, ANSWER_VERSION, answer_call),
        critique,
        rewrite,
    )
