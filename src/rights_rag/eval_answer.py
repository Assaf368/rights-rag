"""Answer-stage evaluation.

Every prompt version answers from the same retrieved sources (structure +
headings, reranked, top 5), so the prompt is the only thing that changes.
Primary metrics are checked in code; an LLM judge (Haiku) is a secondary,
non-deterministic metric reported separately. Raw answers and judge labels are
saved, so scores can be recomputed without calling any model.

    uv run python -m rights_rag.eval_answer --versions v0 v1 --judge
    uv run python -m rights_rag.eval_answer --rescore eval/runs/answers_<file>.json
"""

import argparse
import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

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
_CITATION = re.compile(r"«([^»]+)»\s*\[(\d+)\]")


def is_abstention(text: str) -> bool:
    opening = normalize(text)[:ABSTAIN_WINDOW]
    return any(p in opening for p in _ABSTAIN_PHRASES)


def keys_found(keys: list[list[str]], text: str) -> bool:
    t = normalize(text)
    return all(any(normalize(k) in t for k in fact) for fact in keys)


def check_citations(text: str, chunks: list[dict]) -> list[dict]:
    """Each «quote» [n] must appear word for word in source n."""
    out = []
    for quote, n in _CITATION.findall(text):
        i = int(n)
        source = chunks[i - 1]["text"] if 1 <= i <= len(chunks) else ""
        out.append(
            {
                "quote": quote,
                "source": i,
                "valid": normalize(quote) in normalize(source),
            }
        )
    return out


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

_LABELS = ("CORRECT", "PARTIAL", "INCORRECT", "ABSTAINED", "HALLUCINATED")


def judge(question: dict, answer_text: str, client) -> str:
    if question["evidence"]:
        facts = "\n".join(f"- {fact[0]}" for fact in question["evidence"])
        prompt = JUDGE_ANSWERABLE.format(
            question=question["question"], facts=facts, answer=answer_text
        )
    else:
        prompt = JUDGE_UNANSWERABLE.format(
            question=question["question"], answer=answer_text
        )
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=1024,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt}],
    )
    reply = "".join(b.text for b in response.content if b.type == "text").upper()
    return next((label for label in _LABELS if label in reply), "UNPARSED")


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


def generate(questions: list[dict], sources: dict, versions: list[str], call) -> dict:
    from rights_rag.answer import answer

    answers = {v: {} for v in versions}
    for v in versions:
        for i, q in enumerate(questions, 1):
            answers[v][q["id"]] = asdict(
                answer(q["question"], sources[q["id"]], v, call)
            )
            print(f"\r{v}: {i}/{len(questions)}", end="", flush=True)
        print()
    return answers


# ---------------------------------------------------------------- score


def score(questions: list[dict], run: dict) -> dict:
    by_id = {q["id"]: q for q in questions}
    report = {}
    for version, answers in run["answers"].items():
        # A saved run may hold questions since dropped from the golden set.
        answers = {qid: a for qid, a in answers.items() if qid in by_id}
        judged = run.get("judge", {}).get(version, {})
        answerable = [qid for qid in answers if by_id[qid]["evidence"]]
        unanswerable = [qid for qid in answers if not by_id[qid]["evidence"]]
        keyed = [qid for qid in answerable if "answer_keys" in by_id[qid]]
        cites = [
            c
            for qid in answers
            for c in check_citations(answers[qid]["text"], run["sources"][qid])
        ]
        cost = (
            sum(
                a["input_tokens"] * PRICES.get(a["model"], (0, 0))[0]
                + a["output_tokens"] * PRICES.get(a["model"], (0, 0))[1]
                for a in answers.values()
            )
            / 1e6
        )
        text = {qid: a["text"] for qid, a in answers.items()}
        r = {
            "abstained_when_unanswerable": _rate(
                [is_abstention(text[q]) for q in unanswerable]
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
                        " ".join(c["text"] for c in run["sources"][q]),
                    )
                    for q in keyed
                ]
            ),
            "answers_with_citations": _rate(
                [bool(_CITATION.search(t)) for t in text.values()]
            ),
            "citations_valid": (sum(c["valid"] for c in cites) / len(cites))
            if cites
            else None,
            "citations_total": len(cites),
            "refusals": sum(a["stop_reason"] == "refusal" for a in answers.values()),
            "cost_usd": round(cost, 3),
            "n": {
                "answerable": len(answerable),
                "unanswerable": len(unanswerable),
                "keyed": len(keyed),
            },
        }
        if judged:
            labels = [judged[q] for q in answerable if q in judged]
            r["judge_answerable"] = {
                lab: labels.count(lab) / len(labels) for lab in sorted(set(labels))
            }
            labels_u = [judged[q] for q in unanswerable if q in judged]
            r["judge_unanswerable"] = {
                lab: labels_u.count(lab) / len(labels_u)
                for lab in sorted(set(labels_u))
            }
        report[version] = r
    return report


def _rate(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def print_report(report: dict) -> None:
    rows = [
        ("abstained, unanswerable (higher is better)", "abstained_when_unanswerable"),
        ("abstained, answerable (lower is better)", "abstained_when_answerable"),
        ("key facts in answer", "key_facts_correct"),
        ("  ...stated though missing from sources", "key_facts_not_in_sources"),
        ("answers with «quote» [n]", "answers_with_citations"),
        ("citations found in their source", "citations_valid"),
    ]
    versions = list(report)
    print(f"{'metric':44}" + "".join(f"{v:>10}" for v in versions))
    for label, key in rows:
        vals = [report[v][key] for v in versions]
        print(
            f"{label:44}"
            + "".join(f"{'-' if x is None else f'{x:.2f}':>10}" for x in vals)
        )
    print(
        f"{'citations checked':44}"
        + "".join(f"{report[v]['citations_total']:>10}" for v in versions)
    )
    print(
        f"{'refusals':44}" + "".join(f"{report[v]['refusals']:>10}" for v in versions)
    )
    print(
        f"{'answer cost ($)':44}"
        + "".join(f"{report[v]['cost_usd']:>10}" for v in versions)
    )
    if any("judge_answerable" in report[v] for v in versions):
        print("\nLLM judge (Haiku, not deterministic):")
        for v in versions:
            print(f"  {v} answerable:   {_fmt(report[v].get('judge_answerable', {}))}")
            print(
                f"  {v} unanswerable: {_fmt(report[v].get('judge_unanswerable', {}))}"
            )
    print("\nn =", report[versions[0]]["n"])


def _fmt(d: dict) -> str:
    return ", ".join(f"{k} {v:.2f}" for k, v in d.items()) or "-"


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--versions", nargs="+", default=["v0", "v1"])
    parser.add_argument(
        "--judge", action="store_true", help="also grade with the Haiku judge"
    )
    parser.add_argument(
        "--rescore", type=Path, help="score a saved run (adds judge labels if --judge)"
    )
    parser.add_argument(
        "--unanswerable-only",
        action="store_true",
        help="run only questions with no answer",
    )
    args = parser.parse_args()

    questions = [
        q for q in load_golden() if args.split == "all" or q["split"] == args.split
    ]
    if args.unanswerable_only:
        questions = [q for q in questions if not q["evidence"]]
    by_id = {q["id"]: q for q in questions}
    if args.rescore:
        path = args.rescore
        run = json.loads(path.read_text(encoding="utf-8"))
    else:
        from rights_rag.answer import claude_caller

        sources = retrieve_sources(questions)
        run = {"split": args.split, "sources": sources}
        run["answers"] = generate(questions, sources, args.versions, claude_caller())
        RUNS.mkdir(parents=True, exist_ok=True)
        tag = "_unanswerable" if args.unanswerable_only else ""
        path = RUNS / f"answers_{args.split}{tag}_{datetime.now():%Y%m%d-%H%M%S}.json"
    if args.judge:
        import anthropic

        client = anthropic.Anthropic()
        for version, answers in run["answers"].items():
            labels = run.setdefault("judge", {}).setdefault(version, {})
            for qid, a in answers.items():
                if qid not in labels and qid in by_id:
                    labels[qid] = judge(by_id[qid], a["text"], client)
    path.write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
    print(f"run saved to {path}\n")
    print_report(score(questions, run))


if __name__ == "__main__":
    main()
