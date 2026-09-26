"""Sign-in: a shared list of accounts, a signed cookie, and a limit on wrong guesses.

Off unless ``KYR_LOGIN_USERS`` names at least one account, so a local run stays open. When on,
every page and endpoint needs the cookie except the sign-in page itself, the health check (for
uptime monitors) and the stylesheet the sign-in page uses. The admin bearer token also passes,
so scripts and ``/api/status`` keep working without a browser.

The cookie is ``base64(user|expiry).hmac``. It is signed with ``KYR_SESSION_SECRET`` when set,
otherwise with a key derived from the account list, so changing any password signs everyone out
and a restart does not. Each account's own password is also mixed into its signature, so removing
or changing one account ends only that account's sessions.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import os
import re
import secrets
import time
from collections import deque
from urllib.parse import parse_qs

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from .. import config

COOKIE = "kyr_session"
OPEN_PATHS = frozenset({"/login", "/logout", "/api/health", "/static/styles.css",
                        "/static/mark.svg", "/static/favicon.svg", "/static/favicon-32.png",
                        "/static/apple-touch-icon.png", "/favicon.ico"})
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")
FAILURE_LIMIT = 10              # wrong passwords per address...
FAILURE_WINDOW_S = 15 * 60      # ...within this many seconds, before sign-in is refused


def parse_users(raw: str) -> dict[str, str]:
    """``"admin:pw1,guest:pw2"`` → ``{"admin": "pw1", "guest": "pw2"}``. Malformed pairs are
    skipped. A password may contain ``:`` but not ``,``."""
    users = {}
    for pair in raw.split(","):
        name, sep, password = pair.strip().partition(":")
        if sep and password and NAME_RE.match(name):
            users[name] = password
    return users


class Attempts:
    """Wrong guesses per address. Past FAILURE_LIMIT within the window the address is locked out
    until its oldest guess ages out. Used for sign-in and for the budget reset code."""

    def __init__(self) -> None:
        self.failures: dict[str, deque[float]] = {}

    def locked_out(self, address: str, now: float | None = None) -> bool:
        now = now or time.time()
        recent = self.failures.get(address)
        while recent and recent[0] < now - FAILURE_WINDOW_S:
            recent.popleft()
        return bool(recent) and len(recent) >= FAILURE_LIMIT

    def record(self, address: str) -> None:
        self.failures.setdefault(address, deque()).append(time.time())
        if len(self.failures) > 10_000:         # a flood of addresses must not grow forever
            self.failures.clear()


def code_matches(supplied: str, expected: str) -> bool:
    """A constant-time comparison; an unset code never matches."""
    return bool(expected) and secrets.compare_digest(supplied.encode(), expected.encode())


class Gate:
    def __init__(self, raw_users: str | None = None, secret: str | None = None,
                 days: float | None = None) -> None:
        raw = os.environ.get("KYR_LOGIN_USERS", "") if raw_users is None else raw_users
        self.users = parse_users(raw)
        secret = os.environ.get("KYR_SESSION_SECRET", "") if secret is None else secret
        self.key = (secret.encode() if secret.strip()
                    else hashlib.sha256(b"kyr-session\0" + raw.encode()).digest())
        self.ttl_s = (days if days is not None else config.LOGIN_DAYS) * 86400
        self.attempts = Attempts()

    @property
    def enabled(self) -> bool:
        return bool(self.users)

    # ── tokens ────────────────────────────────────────────────────────────────────────
    def _sign(self, payload: bytes, user: str) -> str:
        secret = hashlib.sha256(self.users.get(user, "").encode()).digest()
        return hmac.new(self.key, payload + b"\0" + secret, hashlib.sha256).hexdigest()

    def issue(self, user: str, now: float | None = None) -> str:
        expires = int((now or time.time()) + self.ttl_s)
        payload = f"{user}|{expires}".encode()
        return base64.urlsafe_b64encode(payload).decode().rstrip("=") + "." + \
            self._sign(payload, user)

    def user_for(self, token: str | None, now: float | None = None) -> str | None:
        """The account a cookie belongs to, or None if it is missing, forged or expired."""
        if not token or "." not in token:
            return None
        body, mac = token.rsplit(".", 1)
        try:
            payload = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
            user, expires = payload.decode().split("|")
            expires_at = int(expires)
        except (ValueError, UnicodeDecodeError):
            return None
        if user not in self.users or expires_at < (now or time.time()):
            return None
        return user if hmac.compare_digest(mac, self._sign(payload, user)) else None

    # ── passwords ─────────────────────────────────────────────────────────────────────
    def locked_out(self, address: str, now: float | None = None) -> bool:
        return self.attempts.locked_out(address, now)

    def check(self, address: str, user: str, password: str) -> bool:
        expected = self.users.get(user)
        # Compare even for an unknown name, so timing does not reveal which names exist.
        ok = secrets.compare_digest(password.encode(), (expected or "\0").encode()) and \
            expected is not None
        if not ok:
            self.attempts.record(address)
        return ok

    def request_user(self, request: Request) -> str | None:
        return self.user_for(request.cookies.get(COOKIE))


def admin_bearer(request: Request) -> bool:
    supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    return bool(config.ADMIN_TOKEN) and secrets.compare_digest(supplied, config.ADMIN_TOKEN)


def is_https(request: Request) -> bool:
    if request.url.scheme == "https":
        return True
    return config.TRUST_PROXY_HEADERS and \
        request.headers.get("x-forwarded-proto", "").lower() == "https"


def refuse(request: Request) -> Response:
    """Not signed in: pages go to the sign-in form, the API gets a 401 the UI understands."""
    if request.url.path.startswith("/api/"):
        return JSONResponse({"error": {"kind": "auth", "message": "Please sign in.",
                                       "retry_after_s": 0}}, status_code=401)
    return RedirectResponse("/login", status_code=303)


async def read_form(request: Request) -> dict[str, str]:
    """A urlencoded form, parsed without python-multipart. Capped at 4 KB."""
    raw = (await request.body())[:4096].decode("utf-8", "replace")
    return {k: v[0] for k, v in parse_qs(raw, keep_blank_values=True).items()}


def login_page(error: str = "", user: str = "", status: int = 200) -> HTMLResponse:
    notice = f'<p class="login-error" role="alert">{html.escape(error)}</p>' if error else ""
    return HTMLResponse(LOGIN_HTML.replace("{{notice}}", notice)
                        .replace("{{user}}", html.escape(user, quote=True)),
                        status_code=status, headers={"Cache-Control": "no-store"})


LOGIN_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sign in — KnowYourRights</title>
  <meta name="robots" content="noindex">
  <link rel="icon" href="/static/favicon.svg" type="image/svg+xml">
  <link rel="apple-touch-icon" href="/static/apple-touch-icon.png">
  <meta name="theme-color" content="#0e6b5a">
  <link rel="stylesheet" href="/static/styles.css">
  <style>
    .login-wrap { min-height: 100vh; display: flex; align-items: center; justify-content: center;
      padding: 16px; }
    .login-card { width: 100%; max-width: 360px; background: var(--surface);
      border: 1px solid var(--border); border-radius: var(--radius); padding: 28px 24px; }
    .login-card .login-logo { display: block; width: 52px; height: 52px; margin: 0 0 14px;
      border-radius: 13px; }
    .login-card h1 { margin: 0 0 4px; font: 600 1.45rem/1.25 var(--serif); }
    .login-card p.sub { margin: 0 0 20px; color: var(--text-dim); font-size: .9rem; }
    .login-card label { display: block; font-size: .85rem; margin: 12px 0 4px; }
    .login-card input { width: 100%; box-sizing: border-box; padding: 9px 10px; font: inherit;
      color: var(--text); background: var(--bg); border: 1px solid var(--border);
      border-radius: 8px; }
    .login-card input:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
    .login-card button { margin-top: 20px; width: 100%; padding: 10px; font: inherit;
      font-weight: 600; color: #fff; background: var(--accent); border: 0; border-radius: 8px;
      cursor: pointer; }
    .login-error { color: var(--danger); border: 1px solid var(--danger); border-radius: 8px;
      padding: 8px 10px; font-size: .9rem; margin: 0 0 8px; }
  </style>
</head>
<body>
<main class="login-wrap">
  <form class="login-card" method="post" action="/login">
    <img class="login-logo" src="/static/mark.svg" alt="">
    <h1>KnowYourRights</h1>
    <p class="sub">Sign in to continue.</p>
    {{notice}}
    <label for="user">Username</label>
    <input id="user" name="user" autocomplete="username" required autofocus value="{{user}}">
    <label for="password">Password</label>
    <input id="password" name="password" type="password" autocomplete="current-password" required>
    <button type="submit">Sign in</button>
  </form>
</main>
</body>
</html>
"""
