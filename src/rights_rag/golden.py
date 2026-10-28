"""The golden question set and its checks.

Each question lists the facts its answer needs. A fact is a list of accepted
spans (the same fact may be worded differently on different pages). Spans are
exact text from the corpus, so they stay valid whatever the chunking.

    uv run python -m rights_rag.golden   # validate eval/golden.jsonl
"""

import json
import re
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document

GOLDEN = Path("eval/golden.jsonl")
TEST_EVERY = 3  # every 3rd question of each type goes to the held-out test split

_INVISIBLE = re.compile(r"[\u200e\u200f\u202a-\u202e]")  # direction marks
_TOKEN = re.compile(r"[\w\"'״]+")
# One-letter prefixes that glue onto Hebrew words (and, the, in, to, from, that, as).
_PREFIXES = "והבלמשכ"


def normalize(text: str) -> str:
    text = _INVISIBLE.sub("", text).replace("־", "-").replace("״", '"')
    return re.sub(r"\s+", " ", text).strip()


def load_golden(path: Path = GOLDEN) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def fact_found(fact: list[str], text: str) -> bool:
    """True if any accepted span of the fact appears in the (normalized) text."""
    return any(normalize(span) in text for span in fact)


def _stems(text: str) -> set[str]:
    stems = set()
    for tok in _TOKEN.findall(text):
        while len(tok) > 3 and tok[0] in _PREFIXES:
            tok = tok[1:]
        if len(tok) >= 3:
            stems.add(tok)
    return stems


def lexical_overlap(question: str, evidence: list[list[str]]) -> float:
    """Share of the question's words that also appear in the evidence: a rough
    check that a question was not copied from its source."""
    q = _stems(question)
    e = (
        set().union(*(_stems(s) for fact in evidence for s in fact))
        if evidence
        else set()
    )
    return len(q & e) / len(q) if q else 0.0


def assign_splits(questions: list[dict]) -> list[dict]:
    """Give questions without a split one, stratified by type. Splits already in
    the file are kept, so adding questions never moves old ones."""
    seen: Counter[str] = Counter()
    for q in questions:
        seen[q["type"]] += 1
        q.setdefault("split", "test" if seen[q["type"]] % TEST_EVERY == 0 else "dev")
    return questions


def validate(questions: list[dict], sections: list[Document]) -> list[str]:
    """Return problems: spans missing from the corpus, duplicate ids."""
    pages: dict[str, str] = {}
    for s in sections:
        pages[s.metadata["doc_id"]] = (
            pages.get(s.metadata["doc_id"], "") + "\n" + s.page_content
        )
    pages = {k: normalize(v) for k, v in pages.items()}

    problems = []
    ids = Counter(q["id"] for q in questions)
    problems += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    for q in questions:
        for fact in q["evidence"]:
            for span in fact:
                if not any(normalize(span) in text for text in pages.values()):
                    problems.append(f"{q['id']}: span not in corpus: {span[:60]}")
        q["source_docs"] = sorted(
            {
                d
                for fact in q["evidence"]
                for d, text in pages.items()
                if fact_found(fact, text)
            }
        )
    return problems


def main() -> None:
    from rights_rag.corpus import load_corpus

    questions = load_golden()
    problems = validate(questions, load_corpus())
    assign_splits(questions)
    for q in questions:
        overlap = lexical_overlap(q["question"], q["evidence"])
        flag = "  <- close to source wording" if overlap > 0.5 else ""
        print(
            f"{q['id']} {q['split']:4} {q['type']:12} overlap={overlap:.2f} docs={q['source_docs']}{flag}"
        )
    print()
    print("types:", dict(Counter(q["type"] for q in questions)))
    print("splits:", dict(Counter(q["split"] for q in questions)))
    if problems:
        raise SystemExit("problems:\n" + "\n".join(problems))
    print("all spans found in the corpus")


if __name__ == "__main__":
    main()
