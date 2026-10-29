"""Progress events for streaming. The pipeline calls emit() at each step; a
request that streams sets a listener for its own run, and everything else
(CLI, evaluation, tests) has no listener, so the events go nowhere."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_listener: ContextVar[Callable[[dict], None] | None] = ContextVar(
    "listener", default=None
)


def emit(type_: str, **data) -> None:
    listener = _listener.get()
    if listener:
        listener({"type": type_, **data})


def active() -> bool:
    """Whether anyone listens; skip building costly event data otherwise."""
    return _listener.get() is not None


@contextmanager
def listening(listener: Callable[[dict], None]) -> Iterator[None]:
    token = _listener.set(listener)
    try:
        yield
    finally:
        _listener.reset(token)


def public_sources(chunks: list[dict]) -> list[dict]:
    """Sources as the API shows them, numbered as the prompt numbers them."""
    return [
        {
            "n": n,
            "title": c["title"],
            "section": c["section_path"],
            "url": c["url"],
            "updated": c.get("updated") or None,
            "text": c["text"],
        }
        for n, c in enumerate(chunks, start=1)
    ]
