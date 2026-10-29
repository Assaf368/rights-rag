"""Write score summaries of the saved runs behind the README's tables.

Raw runs (eval/runs/) hold text from the source pages, so they stay out of
the repository. This re-scores each one, without calling any model, and
writes only the numbers to eval/results/, which is committed.

    uv run python -m rights_rag.results
"""

import io
import json
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

from rights_rag import eval_answer, eval_retrieval
from rights_rag.eval_critic import correct
from rights_rag.golden import load_golden

RUNS = Path("eval/runs")
OUT = Path("eval/results")

# README section -> the saved run it reports, in README order.
RETRIEVAL = [
    ("1-chunking-dev", "retrieval_dev_20261028-113215.json"),
    ("1-chunking-test", "chunking_test_20261028-122344.json"),
    ("2-hybrid-dev", "hybrid_dev_20261028-120952.json"),
    ("3-rerank-dev", "rerank_dev_20261028-122009.json"),
    ("3-rerank-test", "rerank_test_20261028-122512.json"),
    ("3-rerank-handwritten", "rerank_assaf_20261029-113227.json"),
    ("7-rewrite-dev", "rewrite_dev_20261029-122450.json"),
    ("7-rewrite-handwritten", "rewrite_assaf_20261029-123537.json"),
    ("10-rerank-hard", "rerank_hard_20261029-152324.json"),
]
ANSWERS = [
    ("4-prompt-v0-v1-dev", "answers_dev_20261028-141012.json"),
    ("4-prompt-v0-v1-unanswerable", "answers_all_unanswerable_20261028-143024.json"),
    ("5-test", "answers_test_20261029-111725.json"),
    ("6-9-answers-dev", "answers_dev_20261028-152311.json"),
    ("9-answers-handwritten", "answers_assaf_20261029-113545.json"),
    ("10-answers-hard", "answers_hard_20261029-152617.json"),
]
CRITIC = [
    ("8-critic-without-page-list", "critic_20261029-125458.json"),
    ("8-critic-with-page-list", "critic_20261029-125738.json"),
]


def _questions(split: str) -> list[dict]:
    return [q for q in load_golden() if split == "all" or q["split"] == split]


def retrieval(name: str) -> str:
    saved = json.loads((RUNS / name).read_text(encoding="utf-8"))
    questions = _questions(saved["split"])
    out = io.StringIO()
    with redirect_stdout(out):
        eval_retrieval.print_report(eval_retrieval.score(questions, saved["run"]))
    return f"split: {saved['split']}\n\n{out.getvalue()}"


def answers(name: str) -> str:
    run = json.loads((RUNS / name).read_text(encoding="utf-8"))
    questions = _questions(run["split"])
    out = io.StringIO()
    with redirect_stdout(out):
        eval_answer.print_report(eval_answer.score(questions, run))
    return f"split: {run['split']}\n\n{out.getvalue()}"


def critic(name: str) -> str:
    results = json.loads((RUNS / name).read_text(encoding="utf-8"))
    lines = []
    for expected in ("retry_with_rewrite", "accept", "keep"):
        rows = [r for r in results if r["expected"] == expected]
        right = sum(correct(expected, r["action"]) for r in rows)
        got = dict(Counter(r["action"] for r in rows))
        lines.append(f"expected {expected:20} right {right}/{len(rows)}  got {got}")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for kind, items in ((retrieval, RETRIEVAL), (answers, ANSWERS), (critic, CRITIC)):
        for title, name in items:
            body = kind(name)
            (OUT / f"{title}.txt").write_text(f"run: {name}\n{body}", encoding="utf-8")
            print(f"wrote {OUT / title}.txt")


if __name__ == "__main__":
    main()
