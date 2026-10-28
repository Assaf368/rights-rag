"""The answer stage: retrieved chunks + a prompt version -> Claude's answer."""

from collections.abc import Callable
from dataclasses import dataclass

import anthropic

from rights_rag.prompts import PROMPTS

ANSWER_MODEL = "claude-sonnet-5-5"
ANSWER_EFFORT = "medium"
MAX_TOKENS = 4000


@dataclass
class Answer:
    text: str
    stop_reason: str
    input_tokens: int
    output_tokens: int
    model: str


def claude_caller(
    client: anthropic.Anthropic | None = None,
) -> Callable[[str | None, str], Answer]:
    """Return a function (system, user) -> Answer that calls Claude.

    Server-side fallback is on: if the model declines on safety grounds, the
    API retries on the recommended fallback model in the same call.
    """
    client = client or anthropic.Anthropic()

    def call(system: str | None, user: str) -> Answer:
        kwargs = {"system": system} if system else {}
        response = client.beta.messages.create(
            model=ANSWER_MODEL,
            max_tokens=MAX_TOKENS,
            output_config={"effort": ANSWER_EFFORT},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
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


def answer(
    question: str,
    chunks: list[dict],
    version: str,
    call: Callable[[str | None, str], Answer],
) -> Answer:
    system, user = PROMPTS[version](question, chunks)
    return call(system, user)
