"""FastAPI app: an SSE chat endpoint, a health probe, and the static UI.

One worker, deliberately. The models are process-resident and a second worker would load
another copy of bge-m3 onto a 4 GB card.

Startup does the expensive things once — opening LanceDB, loading and *warming* both models —
so the first question does not pay the 1.9s CUDA warmup on top of everything else.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, events
from .context.memory import Conversation
from .llm.client import get_client
from .llm.ledger import get_ledger
from .orchestrator import get_orchestrator
from .retrieval.search import get_engine
from .runtime import gpu, resources
from .runtime.cache import get_cache
from .tools import crawl

log = logging.getLogger(__name__)

# Playwright drives Chromium through a subprocess transport that the Windows selector event
# loop cannot host. Without this, every browser-backed crawl fails under uvicorn.
if sys.platform == "win32":
    with contextlib.suppress(Exception):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str = ""
    depth: str = "auto"
    state: str = ""


class FeedbackRequest(BaseModel):
    session_id: str = ""
    rating: str = ""
    question: str = ""
    answer: str = ""
    comment: str = ""


SESSIONS: dict[str, Conversation] = {}
SESSION_LAST_SEEN: dict[str, float] = {}
MAX_SESSIONS = 200
SESSION_TTL_S = 6 * 3600


def get_conversation(session_id: str) -> Conversation:
    session_id = (session_id or "").strip() or uuid.uuid4().hex[:16]
    now = time.time()
    stale = [sid for sid, seen in SESSION_LAST_SEEN.items() if now - seen > SESSION_TTL_S]
    for sid in stale:
        SESSIONS.pop(sid, None)
        SESSION_LAST_SEEN.pop(sid, None)
    while len(SESSIONS) > MAX_SESSIONS:
        oldest = min(SESSION_LAST_SEEN, key=SESSION_LAST_SEEN.get)
        SESSIONS.pop(oldest, None)
        SESSION_LAST_SEEN.pop(oldest, None)

    conversation = SESSIONS.get(session_id)
    if conversation is None:
        conversation = Conversation(session_id=session_id)
        SESSIONS[session_id] = conversation
    SESSION_LAST_SEEN[session_id] = now
    return conversation


app = FastAPI(title="KnowYourRights", docs_url=None, redoc_url=None)
_ready = {"models": False, "error": "", "detail": {}}


@app.on_event("startup")
async def startup() -> None:
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config.ensure_runtime_dirs()
    plan = resources.get_plan()
    log.info("resource plan:\n%s", plan.describe())

    async def warm() -> None:
        try:
            detail = await get_engine().warmup()
            _ready["models"] = detail["embedder"]["loaded"] or bool(detail["store"])
            _ready["detail"] = detail
            log.info("ready — ask questions at http://%s:%s", config.HOST, config.PORT)
        except Exception as exc:
            _ready["error"] = str(exc)
            log.exception("warmup failed; retrieval will run degraded")

    # Warm in the background so the port is listening immediately.
    asyncio.create_task(warm())
    if config.MODEL_IDLE_EVICT_S > 0:
        asyncio.create_task(gpu.evict_loop())
    asyncio.create_task(_browser_reaper())


async def _browser_reaper() -> None:
    """Hand back Chromium's 300-500 MB when nobody is crawling."""
    while True:
        try:
            await asyncio.sleep(60)
            await crawl.get_crawler().close_browser_if_idle()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.debug("browser reaper: %s", exc)


@app.on_event("shutdown")
async def shutdown() -> None:
    with contextlib.suppress(Exception):
        await crawl.get_crawler().aclose()
    with contextlib.suppress(Exception):
        await get_client().aclose()
    gpu.get_executor().shutdown()


@app.post("/api/chat")
async def chat(request: ChatRequest, http_request: Request) -> StreamingResponse:
    conversation = get_conversation(request.session_id)
    orchestrator = get_orchestrator()
    depth = request.depth if request.depth in config.DEPTHS else None

    async def event_stream():
        # Tell the client its session id first so reconnects land in the same conversation.
        yield events.Event("session", {"session_id": conversation.session_id}).to_sse()
        try:
            async for event in orchestrator.stream(
                request.message, conversation, depth=depth,
                state=request.state.strip() or None,
            ):
                if await http_request.is_disconnected():
                    log.info("client disconnected; cancelling turn")
                    orchestrator.cancel(conversation.session_id)
                    break
                yield event.to_sse()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("stream failed")
            yield events.error(f"Server error: {str(exc)[:200]}").to_sse()
            yield events.done().to_sse()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # keep proxies from buffering the stream
        },
    )


@app.post("/api/stop")
async def stop(payload: dict) -> dict:
    session_id = (payload or {}).get("session_id", "")
    return {"cancelled": get_orchestrator().cancel(session_id)}


@app.post("/api/reset")
async def reset(payload: dict) -> dict:
    """Start over: drop the conversation entirely rather than just emptying it.

    Turns, the summary and the evidence pool all go, and so does the session entry itself —
    the pool holds full section text and crawled pages, so dropping it is the single biggest
    thing a long-running server can do to give memory back.
    """
    session_id = (payload or {}).get("session_id", "")
    conversation = SESSIONS.pop(session_id, None)
    SESSION_LAST_SEEN.pop(session_id, None)
    if conversation:
        conversation.reset()
    return {"ok": True, "cleared": bool(conversation)}


@app.post("/api/feedback")
async def feedback(request: FeedbackRequest) -> dict:
    """Thumbs go to a JSONL file that feeds the eval set."""
    import json

    config.ensure_runtime_dirs()
    row = {"at": time.time(), **request.model_dump()}
    try:
        with open(config.RUNTIME_DIR / "feedback.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        log.warning("could not record feedback: %s", exc)
    return {"ok": True}


@app.get("/api/health")
async def health() -> JSONResponse:
    plan = resources.get_plan()
    engine = get_engine()
    payload = {
        "ready": _ready["models"],
        "warmup_error": _ready["error"],
        "profile": plan.name,
        "resources": resources.live_usage(),
        "retrieval": engine.status(),
        "nim": get_client().status(),
        "crawler": crawl.get_crawler().status(),
        "cache": get_cache().stats(),
        "sessions": {
            "active": len(SESSIONS),
            "max": MAX_SESSIONS,
            "ttl_hours": SESSION_TTL_S // 3600,
            "pooled_sources": sum(len(c.pool) for c in SESSIONS.values()),
        },
        "depths": {name: {"rounds": b.max_rounds, "crawls": b.max_crawls,
                          "deadline_s": b.deadline_s}
                   for name, b in config.DEPTHS.items()},
        "states": list(config.INDIAN_STATES),
        "disclaimer": config.DISCLAIMER,
    }
    return JSONResponse(payload)


@app.get("/api/usage")
async def usage() -> dict:
    return get_ledger().snapshot()


# ── static UI ─────────────────────────────────────────────────────────────────────────
if config.WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(config.WEB_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    page = config.WEB_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(500, "UI not found — knowyourrights/web/index.html is missing")
    return FileResponse(page)


def main() -> None:
    import uvicorn

    uvicorn.run("knowyourrights.server:app", host=config.HOST, port=config.PORT,
                workers=1, log_level=config.LOG_LEVEL.lower(),
                reload=bool(os.environ.get("KYR_RELOAD")))


if __name__ == "__main__":
    main()
