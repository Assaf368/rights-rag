"""REST API.

    uv run uvicorn rights_rag.api:app --reload
    curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
         -d '{"question": "כמה ימי חופשה מגיעים לי?"}'
"""

import asyncio
import json
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rights_rag.service import Answerer, default_answerer

WEB_ORIGINS = ["http://localhost:5173"]  # the Vite dev server


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    critic: bool = True


def create_app(make_answerer: Callable[[], Answerer] = default_answerer) -> FastAPI:
    """make_answerer is injected so tests can pass one with fake models."""
    state = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["answerer"] = make_answerer()  # loads the models once
        yield

    app = FastAPI(title="Rights RAG", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=WEB_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"ok": "answerer" in state}

    @app.post("/ask")
    async def ask(req: AskRequest) -> dict:
        # The pipeline is blocking (models and API calls); keep the event loop free.
        return await run_in_threadpool(state["answerer"].ask, req.question, req.critic)

    @app.post("/ask/stream")
    async def ask_stream(req: AskRequest) -> StreamingResponse:
        """Server-sent events: stage, sources, token, retry, blocked, critic,
        replace, then done (with the same response as /ask) or error."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict] = asyncio.Queue()

        def on_event(event: dict) -> None:  # called from the worker thread
            loop.call_soon_threadsafe(queue.put_nowait, event)

        async def events():
            task = loop.run_in_executor(
                None,
                state["answerer"].ask_streaming,
                req.question,
                req.critic,
                on_event,
            )
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event["type"] in ("done", "error"):
                    break
            await task

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()
