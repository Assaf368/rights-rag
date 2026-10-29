"""Critic evaluation on saved answers: does the critic pick the right action?

The expected action comes from the labels: a flagged answerable answer that
misses its key facts should be retried; one that has them should be accepted;
an abstention on an unanswerable question should be kept (accept or abstain).
Good answers that were never flagged are added to check the critic does not
send correct answers back.

    uv run python -m rights_rag.eval_critic
"""

import json
import random
from collections import Counter
from datetime import datetime

from rights_rag.critic import CRITIC_MODEL, critique, flag
from rights_rag.eval_answer import RUNS, keys_found
from rights_rag.golden import load_golden

SAVED = [
    (
        "answers_dev_20261028-152311.json",
        ["v2@claude-haiku-5-5", "v2.1@claude-haiku-5-5"],
    ),
    ("answers_assaf_20261029-113545.json", ["v2@claude-haiku-5-5"]),
]
UNFLAGGED_SAMPLE = 10


def cases() -> list[dict]:
    questions = {q["id"]: q for q in load_golden()}
    out, unflagged = [], []
    for name, versions in SAVED:
        run = json.loads((RUNS / name).read_text(encoding="utf-8"))
        for v in versions:
            for qid, a in run["answers"][v].items():
                q = questions.get(qid)
                if not q:
                    continue
                case = {
                    "run": name,
                    "version": v,
                    "id": qid,
                    "answer": a["text"],
                    "sources": run["sources"][qid],
                    "question": q["question"],
                }
                reason = flag(a)
                if not q["evidence"]:
                    if reason == "abstained":
                        out.append(case | {"reason": reason, "expected": "keep"})
                elif "answer_keys" in q:
                    ok = keys_found(q["answer_keys"], a["text"])
                    if reason:
                        out.append(
                            case
                            | {
                                "reason": reason,
                                "expected": "accept" if ok else "retry_with_rewrite",
                            }
                        )
                    elif ok:
                        unflagged.append(
                            case | {"reason": "retried", "expected": "accept"}
                        )
    # Unflagged answers are shown with a made-up flag, to tempt the critic.
    return out + random.Random(0).sample(unflagged, UNFLAGGED_SAMPLE)


def correct(expected: str, action: str) -> bool:
    return action in ("accept", "abstain") if expected == "keep" else action == expected


def main() -> None:
    from dotenv import load_dotenv

    from rights_rag.answer import claude_caller

    load_dotenv()
    call = claude_caller(model=CRITIC_MODEL)
    results = []
    todo = cases()
    for i, c in enumerate(todo, 1):
        verdict = critique(c["question"], c["sources"], c["answer"], c["reason"], call)
        results.append(
            {k: c[k] for k in ("run", "version", "id", "reason", "expected")} | verdict
        )
        print(f"\rcritic: {i}/{len(todo)}", end="", flush=True)
    print()
    path = RUNS / f"critic_{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    print(f"saved to {path}\n")
    for expected in ("retry_with_rewrite", "accept", "keep"):
        rows = [r for r in results if r["expected"] == expected]
        got = Counter(r["action"] for r in rows)
        right = sum(correct(expected, r["action"]) for r in rows)
        print(
            f"expected {expected:20} n={len(rows):3}  right {right}/{len(rows)}  got {dict(got)}"
        )
    for r in results:
        if not correct(r["expected"], r["action"]):
            print(
                f"  wrong: {r['id']} {r['version'][:4]} expected {r['expected']}, got {r['action']}: {r['reason']}"
            )


if __name__ == "__main__":
    main()
