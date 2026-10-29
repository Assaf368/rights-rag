"""The answer stage: retrieved chunks + a prompt version -> Claude's answer."""

import re
from collections.abc import Callable
from dataclasses import dataclass

import anthropic

from rights_rag.golden import normalize
from rights_rag.prompts import ABSTAIN_V1, PROMPTS, UNVERIFIED_V2, render_retry_v2

# Haiku answered dev at 1/12 of Sonnet's cost; see the README for the trade-off.
ANSWER_MODEL = "claude-haiku-5-5"
ANSWER_EFFORT = "medium"
# Haiku 5.5 has no server-side fallback; sending one is a 400.
FALLBACK_MODELS = {"claude-sonnet-5-5"}
MAX_TOKENS = 4000


@dataclass
class Answer:
    text: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    model: str
    attempts: int = 1


def claude_caller(
    client: anthropic.Anthropic | None = None,
    model: str = ANSWER_MODEL,
) -> Callable[[str | None, str], Answer]:
    """Return a function (system, user) -> Answer that calls Claude.

    Server-side fallback is on where the model supports it: if the model
    declines on safety grounds, the API retries on the recommended fallback
    model in the same call.
    """
    client = client or anthropic.Anthropic()

    def call(system: str | None, user: str) -> Answer:
        kwargs = {"system": system} if system else {}
        if model in FALLBACK_MODELS:
            kwargs |= {
                "betas": ["server-side-fallback-2026-07-01"],
                "fallbacks": "default",
            }
        response = client.beta.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            output_config={"effort": ANSWER_EFFORT},
            messages=[{"role": "user", "content": user}],
            **kwargs,
        )
        text = (
            ""
            if response.stop_reason == "refusal"
            else "".join(b.text for b in response.content if b.type == "text")
        )
        return Answer(
            text=text,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
        )

    return call


_CITATION = re.compile(r"«([^»]+)»\s*\[(\d+)\]")


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


def verified(text: str, chunks: list[dict]) -> bool:
    """The v2 gate: every quote is in its source, and an answer with no quote
    is the abstention sentence."""
    cites = check_citations(text, chunks)
    if not cites:
        return normalize(ABSTAIN_V1) in normalize(text)
    return all(c["valid"] for c in cites)


def enforce_abstention(text: str) -> str:
    """An answer that contains the abstention sentence is only that sentence."""
    return ABSTAIN_V1 if normalize(ABSTAIN_V1) in normalize(text) else text


# Gated versions and the prompt each one answers with. v2.1 also enforces the
# abstention sentence.
GATED = {"v2": "v1.1", "v2.1": "v1.2"}
ENFORCED = {"v2.1"}


def answer(
    question: str,
    chunks: list[dict],
    version: str,
    call: Callable[[str | None, str], Answer],
) -> Answer:
    if version in GATED:
        return answer_v2(question, chunks, call, version)
    system, user = PROMPTS[version](question, chunks)
    return call(system, user)


def answer_v2(
    question: str,
    chunks: list[dict],
    call: Callable[[str | None, str], Answer],
    version: str = "v2",
) -> Answer:
    """The version's prompt behind the gate, with one retry; an answer that
    still fails is replaced by UNVERIFIED_V2."""
    system, user = PROMPTS[GATED[version]](question, chunks)
    first = call(system, user)
    if version in ENFORCED:
        first.text = enforce_abstention(first.text)
    if verified(first.text, chunks):
        return first
    failed = [c for c in check_citations(first.text, chunks) if not c["valid"]]
    second = call(system, render_retry_v2(user, first.text, failed))
    if version in ENFORCED:
        second.text = enforce_abstention(second.text)
    text = second.text if verified(second.text, chunks) else UNVERIFIED_V2
    return Answer(
        text=text,
        stop_reason=second.stop_reason,
        input_tokens=first.input_tokens + second.input_tokens,
        output_tokens=first.output_tokens + second.output_tokens,
        model=second.model,
        attempts=2,
    )
