"""Answer-stage evaluation.

Every prompt version answers from the same retrieved sources (structure +
headings, reranked, top 5), so the prompt is the only thing that changes.
Primary metrics are checked in code; an LLM judge (Haiku) is a secondary,
non-deterministic metric reported separately. Raw answers and judge labels are
saved, so scores can be recomputed without calling any model.

    uv run python -m rights_rag.eval_answer --versions v0 v1 --judge
    uv run python -m rights_rag.eval_answer --rescore eval/runs/answers_<file>.json
    uv run python -m rights_rag.eval_answer --extend eval/runs/answers_<file>.json \
        --versions v1.1 --model claude-haiku-5-5

Answers are saved under "<version>@<model>", for example
"v1.1@claude-haiku-5-5". Runs before the switch to Haiku used the bare version
for Sonnet answers.
"""

import argparse
import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from rights_rag.answer import _CITATION, check_citations, verified
from rights_rag.golden import load_golden, normalize
from rights_rag.prompts import ABSTAIN_V1

RUNS = Path("eval/runs")
COLLECTION = "structure_headings__e5-large"
RERANK_DEPTH = 20
TOP_K = 5
JUDGE_MODEL = "claude-haiku-5-5"
# Sonnet 5.5 and Haiku 5.5 prices, $ per million tokens (input, output).
PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}

# v1 has an exact sentence; v0 does not, so common Hebrew ways of saying
# "the sources don't cover this" also count. Only the opening counts: answers
# often end with a "what is missing" note, which is not an abstention.
_ABSTAIN_PHRASES = (
    ABSTAIN_V1[:-1],
    "אין במקורות",
    "אין מידע",
    "אין בהקשר",
    "אין התייחסות",
    "אין לי בסיס",
    "לא מופיע",
    "לא מצוין",
    "לא נמצא מידע",
    "אינם מתייחסים",
    "אינו מתייחס",
    "אינו עוסק",
    "אינם עוסקים",
    "אינו מגדיר",
    "אינו כולל",
    "אינם כוללים",
    "אי אפשר לענות",
    "לא ניתן לענות",
    "לא ניתן לדעת",
)
ABSTAIN_WINDOW = 150  # characters from the start of the answer


def is_abstention(text: str) -> bool:
    opening = normalize(text)[:ABSTAIN_WINDOW]
    return any(p in opening for p in _ABSTAIN_PHRASES)


def keys_found(keys: list[list[str]], text: str) -> bool:
    t = normalize(text)
    return all(any(normalize(k) in t for k in fact) for fact in keys)


# ---------------------------------------------------------------- judge

JUDGE_ANSWERABLE = """You grade an answer to a question about Israeli employment rights.
The reference facts below are copied from the source and are correct.

Labels:
CORRECT - the answer gives the substance of all reference facts and nothing that contradicts them.
PARTIAL - the answer gives some of the facts, or omits a condition that changes the answer.
INCORRECT - the answer contradicts the facts or gives a wrong number.
ABSTAINED - the answer says the information is not available.

Question: {question}

Reference facts:
{facts}

Answer to grade:
{answer}

Reply with one label only."""

JUDGE_UNANSWERABLE = """You grade an answer to a question about Israeli employment rights.
The sources the system had do NOT contain the answer to this question.

Labels:
ABSTAINED - the answer says it does not have the information (it may suggest where to look).
HALLUCINATED - the answer gives a substantive answer anyway.

Question: {question}

Answer to grade:
{answer}

Reply with one label only."""

JUDGE_V2_ANSWERABLE = """You grade an answer from a retrieval system about Israeli employment rights.
The system saw only the numbered sources below. The reference facts are correct
and come from the corpus; they may or may not be among these sources.

Labels:
CORRECT - the answer gives the substance of the reference facts that the sources
  contain, and every claim it makes is supported by the sources.
PARTIAL - the answer misses a reference fact that the sources do contain, or omits
  a condition from the sources that changes the answer.
UNSUPPORTED - the answer makes a claim the sources do not support, even if the
  claim is true (for example, a number from general knowledge).
INCORRECT - the answer contradicts the reference facts or the sources.
ABSTAINED - the answer says it cannot answer.

Saying that the sources do not cover a detail is not a mistake when that is true.

Question: {question}

Reference facts:
{facts}

Sources:
{sources}

Answer to grade:
{answer}

Reply with one label only."""

JUDGE_V2_UNANSWERABLE = """You grade an answer from a retrieval system about Israeli employment rights.
The system saw only the numbered sources below, and they do not answer the question.

Labels:
ABSTAINED - the answer says it cannot answer the question from its sources. It may
  quote related material from the sources or suggest where to look.
HALLUCINATED - the answer gives a substantive answer to the question that the
  sources do not support.

Question: {question}

Sources:
{sources}

Answer to grade:
{answer}

Reply with one label only."""

# INCORRECT contains CORRECT, so labels are matched as whole words.
_LABELS = (
    "INCORRECT",
    "UNSUPPORTED",
    "HALLUCINATED",
    "ABSTAINED",
    "PARTIAL",
    "CORRECT",
)
_LABEL = re.compile(r"\b(" + "|".join(_LABELS) + r")\b")


def judge(
    question: dict, answer_text: str, client, sources: list[dict] | None = None
) -> str:
    """Grade one answer. With sources (judge v2) the judge also checks that every
    claim is supported by what the system saw; without them (judge v1) it only
    compares the answer with the reference facts."""
    facts = "\n".join(f"- {fact[0]}" for fact in question["evidence"])
    if sources is None:
        template = JUDGE_ANSWERABLE if question["evidence"] else JUDGE_UNANSWERABLE
        prompt = template.format(
            question=question["question"], facts=facts, answer=answer_text
        )
    else:
        numbered = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(sources, 1))
        template = (
            JUDGE_V2_ANSWERABLE if question["evidence"] else JUDGE_V2_UNANSWERABLE
        )
        prompt = template.format(
            question=question["question"],
            facts=facts,
            sources=numbered,
            answer=answer_text,
        )
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=1024,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt}],
    )
    reply = "".join(b.text for b in response.content if b.type == "text").upper()
    match = _LABEL.search(reply)
    return match.group(1) if match else "UNPARSED"


# ---------------------------------------------------------------- run


def retrieve_sources(questions: list[dict]) -> dict[str, list[dict]]:
    from rights_rag.index import E5Embedder, open_collection
    from rights_rag.retrieval import Reranker, Retriever

    embedder = E5Embedder()
    retriever = Retriever(open_collection(COLLECTION), embedder.query)
    reranker = Reranker()
    sources = {}
    for q in questions:
        ids = retriever.vector(q["question"], n=RERANK_DEPTH)
        ranked = reranker.rerank(
            q["question"], [(c, retriever.chunks[c]["embed_text"]) for c in ids]
        )
        sources[q["id"]] = [
            {k: v for k, v in hit.items() if k != "embed_text"}
            for hit in retriever.hits(ranked, TOP_K)
        ]
    return sources


def generate(
    questions: list[dict],
    sources: dict,
    versions: list[str],
    call,
    label=str,
    model: str | None = None,
) -> dict:
    """Answer every question with every prompt version; keys are label(version).
    v3 runs the graph from the saved sources and saves its trace with the
    answer, including the sources of a second retrieval."""
    from rights_rag.answer import answer

    answers = {}
    for v in versions:
        out = answers[label(v)] = {}
        if v == "v3":
            from rights_rag.graph import default_graph
            from rights_rag.graph import run as run_graph

            graph = default_graph(model)
        for i, q in enumerate(questions, 1):
            if v == "v3":
                state = run_graph(graph, q["question"], sources[q["id"]])
                out[q["id"]] = state["answer"] | {
                    k: state.get(k) for k in ("flag", "decision", "queries", "calls")
                }
                if state.get("rounds"):
                    out[q["id"]]["sources"] = state["sources"]
            else:
                out[q["id"]] = asdict(answer(q["question"], sources[q["id"]], v, call))
            print(f"\r{label(v)}: {i}/{len(questions)}", end="", flush=True)
        print()
    return answers


# ---------------------------------------------------------------- score


def score(questions: list[dict], run: dict) -> dict:
    by_id = {q["id"]: q for q in questions}
    report = {}
    for version, answers in run["answers"].items():
        # A saved run may hold questions since dropped from the golden set.
        answers = {qid: a for qid, a in answers.items() if qid in by_id}
        answerable = [qid for qid in answers if by_id[qid]["evidence"]]
        unanswerable = [qid for qid in answers if not by_id[qid]["evidence"]]
        keyed = [qid for qid in answerable if "answer_keys" in by_id[qid]]
        cites = [
            c
            for qid in answers
            for c in check_citations(answers[qid]["text"], _sources(run, answers, qid))
        ]
        cost = sum(_cost(c) for a in answers.values() for c in a.get("calls") or [a])
        text = {qid: a["text"] for qid, a in answers.items()}
        passed = {q: verified(text[q], _sources(run, answers, q)) for q in answers}
        r = {
            "abstained_when_unanswerable": _rate(
                [is_abstention(text[q]) for q in unanswerable]
            ),
            "exact_abstention_unanswerable": _rate(
                [normalize(ABSTAIN_V1) in normalize(text[q]) for q in unanswerable]
            ),
            "abstained_when_answerable": _rate(
                [is_abstention(text[q]) for q in answerable]
            ),
            "key_facts_correct": _rate(
                [keys_found(by_id[q]["answer_keys"], text[q]) for q in keyed]
            ),
            "key_facts_not_in_sources": _rate(
                [
                    keys_found(by_id[q]["answer_keys"], text[q])
                    and not keys_found(
                        by_id[q]["answer_keys"],
                        " ".join(c["text"] for c in _sources(run, answers, q)),
                    )
                    for q in keyed
                ]
            ),
            # Split by kind: a bare abstention should carry no quote.
            "answers_with_citations": _rate(
                [bool(_CITATION.search(text[q])) for q in answerable]
            ),
            "abstentions_with_citations": _rate(
                [bool(_CITATION.search(text[q])) for q in unanswerable]
            ),
            "citations_valid": (sum(c["valid"] for c in cites) / len(cites))
            if cites
            else None,
            "citations_total": len(cites),
            # v2 gate, applied to the saved answers (no model call): a blocked
            # answer counts as a miss when answerable, as a non-answer otherwise.
            "gate_blocked_answerable": _rate([not passed[q] for q in answerable]),
            "gate_key_facts": _rate(
                [
                    passed[q] and keys_found(by_id[q]["answer_keys"], text[q])
                    for q in keyed
                ]
            ),
            "gate_safe_unanswerable": _rate(
                [is_abstention(text[q]) or not passed[q] for q in unanswerable]
            ),
            "avg_chars_answerable": _mean([len(text[q]) for q in answerable]),
            "avg_chars_unanswerable": _mean([len(text[q]) for q in unanswerable]),
            "reviewed": _rate([bool(a.get("decision")) for a in answers.values()]),
            "searched_again": _rate(
                [len(a.get("queries") or []) > 1 for a in answers.values()]
            ),
            "retried": _rate([a.get("attempts", 1) > 1 for a in answers.values()]),
            "refusals": sum(a["stop_reason"] == "refusal" for a in answers.values()),
            "cost_usd": round(cost, 3),
            "n": {
                "answerable": len(answerable),
                "unanswerable": len(unanswerable),
                "keyed": len(keyed),
            },
        }
        for name in ("judge", "judge_v2"):
            labels = run.get(name, {}).get(version, {})
            if labels:
                r[name] = _label_rates(labels, answerable, unanswerable)
                # Sanity check of the judge against the deterministic keys.
                pairs = [
                    (
                        labels[q] == "CORRECT",
                        keys_found(by_id[q]["answer_keys"], text[q]),
                    )
                    for q in keyed
                    if q in labels
                ]
                r[name]["agrees_with_keys"] = _rate([a == b for a, b in pairs])
        report[version] = r
    return report


def _sources(run: dict, answers: dict, qid: str) -> list[dict]:
    """The sources an answer was written from: its own after a second
    retrieval (v3), the run's shared ones otherwise."""
    return answers[qid].get("sources") or run["sources"][qid]


def _cost(usage: dict) -> float:
    price_in, price_out = PRICES.get(usage["model"], (0, 0))
    return (usage["input_tokens"] * price_in + usage["output_tokens"] * price_out) / 1e6


def _label_rates(labels: dict, answerable: list, unanswerable: list) -> dict:
    out = {}
    for group, ids in (("answerable", answerable), ("unanswerable", unanswerable)):
        got = [labels[q] for q in ids if q in labels]
        out[group] = (
            {lab: got.count(lab) / len(got) for lab in sorted(set(got))} if got else {}
        )
    return out


def _rate(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def _mean(values: list[int]) -> int | None:
    return round(sum(values) / len(values)) if values else None


def print_report(report: dict) -> None:
    rows = [
        ("abstained, unanswerable (higher is better)", "abstained_when_unanswerable"),
        ("  ...with the exact abstention sentence", "exact_abstention_unanswerable"),
        ("abstained, answerable (lower is better)", "abstained_when_answerable"),
        ("key facts in answer", "key_facts_correct"),
        ("  ...stated though missing from sources", "key_facts_not_in_sources"),
        ("answerable: answers with «quote» [n]", "answers_with_citations"),
        ("unanswerable: answers with «quote» [n]", "abstentions_with_citations"),
        ("citations found in their source", "citations_valid"),
        ("v2 gate: blocked, answerable (lower is better)", "gate_blocked_answerable"),
        ("v2 gate: key facts after the gate", "gate_key_facts"),
        ("v2 gate: abstained or blocked, unanswerable", "gate_safe_unanswerable"),
    ]
    versions = list(report)
    w = max(10, *(len(v) + 2 for v in versions))
    print(f"{'metric':44}" + "".join(f"{v:>{w}}" for v in versions))
    for label, key in rows:
        vals = [report[v][key] for v in versions]
        print(
            f"{label:44}"
            + "".join(f"{'-' if x is None else f'{x:.2f}':>{w}}" for x in vals)
        )
    print(
        f"{'citations checked':44}"
        + "".join(f"{report[v]['citations_total']:>{w}}" for v in versions)
    )
    for label, key in (
        ("answer length, answerable (chars)", "avg_chars_answerable"),
        ("answer length, unanswerable (chars)", "avg_chars_unanswerable"),
    ):
        print(
            f"{label:44}"
            + "".join(
                f"{'-' if report[v][key] is None else report[v][key]:>{w}}"
                for v in versions
            )
        )
    for title, key in (
        ("sent to the critic (v3)", "reviewed"),
        ("searched again after the critic (v3)", "searched_again"),
    ):
        print(f"{title:44}" + "".join(f"{report[v][key]:>{w}.2f}" for v in versions))
    print(
        f"{'retried after a failed gate (v2)':44}"
        + "".join(f"{report[v]['retried']:>{w}.2f}" for v in versions)
    )
    print(
        f"{'refusals':44}" + "".join(f"{report[v]['refusals']:>{w}}" for v in versions)
    )
    print(
        f"{'answer cost ($)':44}"
        + "".join(f"{report[v]['cost_usd']:>{w}}" for v in versions)
    )
    for name, title in (
        ("judge", "judge v1 (Haiku, reference facts only)"),
        ("judge_v2", "judge v2 (Haiku, also sees the sources)"),
    ):
        if any(name in report[v] for v in versions):
            print(f"\n{title}, not deterministic:")
            for v in versions:
                j = report[v].get(name, {})
                print(f"  {v} answerable:   {_fmt(j.get('answerable', {}))}")
                print(f"  {v} unanswerable: {_fmt(j.get('unanswerable', {}))}")
                agree = j.get("agrees_with_keys")
                if agree is not None:
                    print(
                        f"  {v} agrees with answer keys on {agree:.0%} of keyed questions"
                    )
    print("\nn =", report[versions[0]]["n"])


def _fmt(d: dict) -> str:
    return ", ".join(f"{k} {v:.2f}" for k, v in d.items()) or "-"


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--split", choices=["dev", "test", "assaf", "all"], default="dev"
    )
    parser.add_argument("--versions", nargs="+", default=["v0", "v1"])
    parser.add_argument(
        "--judge", action="store_true", help="also grade with the Haiku judge"
    )
    parser.add_argument(
        "--judge-version",
        choices=["1", "2", "both"],
        default="2",
        help="1: reference facts only; 2: also sees the sources",
    )
    parser.add_argument(
        "--rejudge",
        action="store_true",
        help="discard saved labels of the chosen judge first",
    )
    parser.add_argument(
        "--rescore", type=Path, help="score a saved run (adds judge labels if --judge)"
    )
    parser.add_argument(
        "--extend",
        type=Path,
        help="add --versions to a saved run, answering from its saved sources",
    )
    parser.add_argument("--model", help="answer model (default: answer.ANSWER_MODEL)")
    parser.add_argument(
        "--unanswerable-only",
        action="store_true",
        help="run only questions with no answer",
    )
    args = parser.parse_args()

    saved_path = args.rescore or args.extend
    if saved_path:
        run = json.loads(saved_path.read_text(encoding="utf-8"))
        args.split = run["split"]  # score the questions the run was made for
    questions = [
        q for q in load_golden() if args.split == "all" or q["split"] == args.split
    ]
    if args.unanswerable_only:
        questions = [q for q in questions if not q["evidence"]]
    by_id = {q["id"]: q for q in questions}
    if args.rescore:
        path = args.rescore
    else:
        from rights_rag.answer import ANSWER_MODEL, claude_caller

        model = args.model or ANSWER_MODEL
        call = claude_caller(model=model)

        def label(v: str) -> str:
            return f"{v}@{model}"

        if args.extend:
            path = args.extend
            # Answer only the questions the saved run has sources for.
            questions = [q for q in questions if q["id"] in run["sources"]]
            run["answers"] |= generate(
                questions, run["sources"], args.versions, call, label, model
            )
        else:
            sources = retrieve_sources(questions)
            run = {"split": args.split, "sources": sources}
            run["answers"] = generate(
                questions, sources, args.versions, call, label, model
            )
            RUNS.mkdir(parents=True, exist_ok=True)
            tag = "_unanswerable" if args.unanswerable_only else ""
            stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
            path = RUNS / f"answers_{args.split}{tag}_{stamp}.json"
    if args.judge:
        import anthropic

        client = anthropic.Anthropic()
        names = {"1": ["judge"], "2": ["judge_v2"], "both": ["judge", "judge_v2"]}
        for name in names[args.judge_version]:
            if args.rejudge:
                run.pop(name, None)
            for version, answers in run["answers"].items():
                labels = run.setdefault(name, {}).setdefault(version, {})
                for qid, a in answers.items():
                    if qid not in labels and qid in by_id:
                        sources = (
                            _sources(run, answers, qid) if name == "judge_v2" else None
                        )
                        labels[qid] = judge(by_id[qid], a["text"], client, sources)
    path.write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
    print(f"run saved to {path}\n")
    print_report(score(questions, run))


if __name__ == "__main__":
    main()
