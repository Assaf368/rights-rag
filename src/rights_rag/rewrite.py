"""Query rewriting: a worker's everyday question becomes search queries in the
terms the Kol Zchut pages use. It feeds retrieval only; the answer model still
sees the user's own question."""

import re
from collections.abc import Callable

from rights_rag.answer import Answer

MAX_QUERIES = 3

REWRITE_SYSTEM = """You turn a question from a worker in Israel into search queries for a
collection of Hebrew pages about employment rights (the Kol Zchut website).

Write 2 or 3 short search queries in Hebrew, one per line. Use the formal terms
such pages use instead of slang (for example "העיפו אותי" becomes "פיטורים",
"הבוס" becomes "המעסיק"). Keep every condition from the question: hourly or
monthly pay, seniority, age, sector, dismissal or resignation. Do not answer
the question and do not add facts or numbers that are not in it. Output only
the queries, with no numbering and no explanation."""

_BULLET = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s*")


def rewrite(question: str, call: Callable[[str | None, str], Answer]) -> list[str]:
    """Up to MAX_QUERIES search queries for the question."""
    text = call(REWRITE_SYSTEM, f"Question: {question}").text
    queries = [_BULLET.sub("", line).strip() for line in text.splitlines()]
    return [q for q in queries if q][:MAX_QUERIES]
