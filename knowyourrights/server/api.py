"""The FastAPI app: a streaming chat endpoint, the static UI, and a few small JSON endpoints.

One worker process. Conversations, the admission queue and the rate limiters live in memory, so a
second worker would split them and double every limit.

Endpoints:
  POST /api/chat      a question, answered as a stream of server-sent events
  POST /api/stop      stop the running answer        POST /api/reset   forget the conversation
  POST /api/feedback  thumbs up or down on an answer
  GET  /api/config    what the UI needs to render     GET  /api/quota   this client's allowance
  GET  /api/health    liveness and readiness          GET  /api/status  full diagnostics (admin)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__, config, events
from ..context.memory import Conversation
from ..llm import retrieval_api
from ..llm.client import get_client
from ..orchestrator import get_orchestrator
from ..retrieval.search import get_engine
from ..runtime.cache import get_cache
from ..tools import crawl
from .admission import AdmissionQueue, ClientBusy, QueueFull, QueueTimeout, Ticket
from .models import ChatRequest, FeedbackRequest, SessionRequest
from .quota import Guard, client_ip
from .sessions import SessionStore

log = logging.getLogger(__name__)

SSE_HEADERS = {"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive",
               "X-Accel-Buffering": "no"}      # keeps proxies from buffering the stream
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": ("default-src 'self'; img-src 'self' data:; "
                                "style-src 'self' 'unsafe-inline'; script-src 'self'; "
                                "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"),
}


@dataclass
class Services:
    sessions: SessionStore = field(default_factory=SessionStore)
    admission: AdmissionQueue = field(default_factory=AdmissionQueue)
    guard: Guard = field(default_factory=Guard)
    warmup: dict = field(default_factory=lambda: {"done": False, "database": None})


# ── lifecycle ─────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    config.ensure_runtime_dirs()
    _log_configuration()
    services = app.state.services = Services()
    # Warm up in the background so the port is listening immediately.
    background = [asyncio.create_task(_warm(services)),
                  asyncio.create_task(_maintenance(services))]
    try:
        yield
    finally:
        for task in background:
            task.cancel()
        await asyncio.gather(*background, return_exceptions=True)
        await _shut_down(services)


def _log_configuration() -> None:
    if not any(config.provider_available(p) for p in config.PROVIDERS):
        log.error("no model provider is configured: set OPENROUTER_API_KEY. Every question "
                  "will be refused until one is.")
    elif not config.OPENROUTER_API_KEY:
        log.warning("OPENROUTER_API_KEY is not set: search runs on keyword matching alone and "
                    "chat uses the NVIDIA fallback models only")
    log.info("KnowYourRights %s — %d answers at a time, $%.2f per client, $%.2f per day",
             __version__, config.MAX_ACTIVE_TURNS, config.CLIENT_BUDGET_USD,
             config.DAILY_BUDGET_USD)


async def _warm(services: Services) -> None:
    try:
        detail = await get_engine().warmup()
        services.warmup["database"] = bool(detail.get("store"))
        log.info("ready — http://%s:%s", config.HOST, config.PORT)
    except Exception as exc:
        services.warmup["database"] = False
        log.exception("warm-up failed: %s", exc)
    finally:
        services.warmup["done"] = True


async def _maintenance(services: Services) -> None:
    """Every minute, hand back an idle browser's 300-500 MB. Every hour, delete expired cache
    entries (they were otherwise only removed when read again) and save the spend book."""
    minute = 0
    while True:
        await asyncio.sleep(60)
        minute += 1
        try:
            await crawl.get_crawler().close_browser_if_idle()
            if minute % 60 == 0:
                purged = await asyncio.to_thread(get_cache().purge_expired)
                services.guard.book.flush()
                log.info("maintenance: %d expired cache entries removed", purged)
        except Exception as exc:
            log.warning("maintenance step failed: %s", exc)


async def _shut_down(services: Services) -> None:
    services.guard.book.flush()
    closers = (get_orchestrator().aclose(), crawl.get_crawler().aclose(),
               get_client().aclose(), retrieval_api.session().aclose())
    for result in await asyncio.gather(*closers, return_exceptions=True):
        if isinstance(result, Exception):
            log.debug("shutdown step failed: %s", result)
    with contextlib.suppress(Exception):
        get_cache().close()


app = FastAPI(title="KnowYourRights", version=__version__, docs_url=None, redoc_url=None,
              openapi_url=None, lifespan=lifespan)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    for header, value in SECURITY_HEADERS.items():
        response.headers.setdefault(header, value)
    return response


@app.exception_handler(RequestValidationError)
async def invalid_request(_request: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else {}
    field_name = ".".join(str(p) for p in first.get("loc", [])[1:]) or "request"
    message = str(first.get("msg", "bad value")).removeprefix("Value error, ")
    return _error(422, "invalid", f"Invalid {field_name}: {message}")


def _error(status: int, kind: str, message: str, retry_after_s: int = 0) -> JSONResponse:
    headers = {"Retry-After": str(retry_after_s)} if retry_after_s else None
    return JSONResponse({"error": {"kind": kind, "message": message,
                                   "retry_after_s": retry_after_s}},
                        status_code=status, headers=headers)


def _services(request: Request) -> Services:
    return request.app.state.services


# ── chat ──────────────────────────────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(body: ChatRequest, request: Request):
    services = _services(request)
    client = services.guard.book.key(client_ip(request))
    verdict = services.guard.check(client)
    if not verdict.allowed:
        status = 429 if verdict.kind == "rate" else 403
        return _error(status, verdict.kind, verdict.message, verdict.retry_after_s)
    try:
        ticket = services.admission.join(client)
    except ClientBusy:
        return _error(429, "busy_client", "You already have a question in progress. Please wait "
                                          "for it to finish.")
    except QueueFull:
        return _error(503, "busy", "The service is at capacity right now. Please try again in a "
                                   "few minutes.", retry_after_s=60)
    conversation = services.sessions.get(body.session_id)
    return StreamingResponse(_turn_events(services, body, request, client, ticket, conversation),
                             media_type="text/event-stream", headers=SSE_HEADERS)


async def _turn_events(services: Services, body: ChatRequest, request: Request, client: str,
                       ticket: Ticket, conversation: Conversation) -> AsyncIterator[str]:
    try:
        yield events.Event("session", {"session_id": conversation.session_id}).to_sse()
        waited = False
        async for position in services.admission.wait(ticket):
            waited = True
            yield events.queued(position).to_sse()
        if waited:
            yield events.queued(0).to_sse()
        async for event in get_orchestrator().stream(
                body.message, conversation, depth=None if body.depth == "auto" else body.depth,
                state=body.state, on_charge=lambda usd: services.guard.book.charge(client, usd)):
            if await request.is_disconnected():
                get_orchestrator().cancel(conversation.session_id)
                return
            yield event.to_sse()
        if services.guard.exhausted(client):
            yield events.limit("client_budget", services.guard.check(client).message).to_sse()
    except QueueTimeout:
        yield events.error("The service stayed busy for too long, so your question was not "
                           "started. Please try again.").to_sse()
        yield events.done().to_sse()
    finally:
        await services.admission.release(ticket)


@app.post("/api/stop")
async def stop(body: SessionRequest) -> dict:
    return {"cancelled": get_orchestrator().cancel(body.session_id)}


@app.post("/api/reset")
async def reset(body: SessionRequest, request: Request) -> dict:
    """Start over: the conversation, its summary and its remembered sources all go."""
    get_orchestrator().cancel(body.session_id)
    return {"ok": True, "cleared": _services(request).sessions.drop(body.session_id)}


@app.post("/api/feedback")
async def feedback(body: FeedbackRequest, request: Request):
    """Thumbs up or down, appended to a JSONL file that feeds the evaluation set."""
    services = _services(request)
    if services.guard.rate.check(services.guard.book.key(client_ip(request))):
        return _error(429, "rate", "Too many requests. Please slow down.")
    row = {"at": time.time(), **body.model_dump()}
    try:
        config.ensure_runtime_dirs()
        with open(config.RUNTIME_DIR / "feedback.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        log.warning("could not record feedback: %s", exc)
    return {"ok": True}


# ── information ───────────────────────────────────────────────────────────────────────
@app.get("/api/config")
async def public_config() -> dict:
    """Everything the UI renders that the server decides."""
    return {
        "version": __version__,
        "states": list(config.INDIAN_STATES),
        "disclaimer": config.DISCLAIMER,
        "depths": {name: {"rounds": d.max_rounds, "pages": d.max_crawls,
                          "deadline_s": d.deadline_s} for name, d in config.DEPTHS.items()},
        "client_budget_usd": config.CLIENT_BUDGET_USD or None,
    }


@app.get("/api/quota")
async def quota(request: Request) -> dict:
    services = _services(request)
    return services.guard.quota(services.guard.book.key(client_ip(request)))


@app.get("/api/health")
async def health(request: Request) -> JSONResponse:
    """``ok``, ``degraded`` (answers, less thoroughly) or ``unavailable`` (cannot answer)."""
    warm = _services(request).warmup
    fatal, degraded = [], []
    if not any(config.provider_available(p) for p in config.PROVIDERS):
        fatal.append("no model provider is configured")
    if warm["database"] is False:
        fatal.append("the legal database could not be opened")
    if not config.OPENROUTER_API_KEY:
        degraded.append("semantic search is off (no OpenRouter key)")
    elif get_engine().embedder.last_error:
        degraded.append("the embedding service is failing")
    status = "unavailable" if fatal else "degraded" if degraded else "ok"
    body = {"status": status, "ready": warm["done"] and not fatal, "problems": fatal + degraded}
    return JSONResponse(body, status_code=503 if fatal else 200)


@app.get("/api/status")
async def status(request: Request) -> dict:
    """Full diagnostics. Needs the admin token, or a request from this machine without one."""
    _require_admin(request)
    services = _services(request)
    return {
        "version": __version__,
        "retrieval": get_engine().status(),
        "retrieval_api": retrieval_api.session().stats(),
        "models": get_client().status(),
        "crawler": crawl.get_crawler().status(),
        "cache": get_cache().stats(),
        "sessions": services.sessions.stats(),
        "queue": services.admission.stats(),
        "spend": services.guard.book.stats(),
    }


def _require_admin(request: Request) -> None:
    if config.ADMIN_TOKEN:
        supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if secrets.compare_digest(supplied, config.ADMIN_TOKEN):
            return
    elif (not config.TRUST_PROXY_HEADERS and "x-forwarded-for" not in request.headers
          and request.client and request.client.host in ("127.0.0.1", "::1")):
        return
    raise HTTPException(status_code=404)


# ── the UI ────────────────────────────────────────────────────────────────────────────
if config.WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(config.WEB_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    page = config.WEB_DIR / "index.html"
    if not page.is_file():
        raise HTTPException(500, "the UI is missing: knowyourrights/web/index.html")
    return FileResponse(page)


def main() -> None:
    import uvicorn

    uvicorn.run("knowyourrights.server:app", host=config.HOST, port=config.PORT, workers=1,
                log_level=config.LOG_LEVEL.lower())
