"""MCP server: lets any agent talk to gemini.google.com via the web interface.

Authentication comes from cookies decrypted out of a local Chrome/Edge
profile (see cookie_extractor.py), so no official Google API key is needed.

Run (stdio, default)::

    uv run gemini-mcp --browser chrome

Run over HTTP (SSE/streamable)::

    uv run gemini-mcp --transport http --port 8900

Tools exposed:

* gemini_send_message        — send a prompt (or continue a conversation),
                               with per-call ``model`` / ``thinking`` overrides
* gemini_models              — selectable model tiers + thinking levels
* gemini_usage               — local usage counters + fallback events
* gemini_list_conversations  — recent conversations from the account
* gemini_read_conversation   — full history of one conversation
* gemini_delete_conversation — delete a conversation
* gemini_status              — cookie/token diagnostics

``--transport http`` additionally serves an OpenAI-compatible REST facade on
the same port (see api.py): POST /v1/chat/completions, GET /v1/models,
/v1/status, /v1/usage, plus an unauthenticated GET /healthz probe.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import hmac
import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from mcp.server import MCPServer

from . import __version__
from . import state
from .cookie_extractor import cookies_for_gemini
from .gemini_client import (
    DEFAULT_METADATA,
    DEFAULT_MODEL,
    DEFAULT_MODEL_FALLBACK_CHAIN,
    DEFAULT_THINKING,
    GENERATE_LOCK,
    MediaRef,
    MODEL_ALIASES,
    MODEL_CATALOG,
    GeminiWebClient,
    GeminiWebError,
    THINKING_LEVELS,
)

log = logging.getLogger("gemini_mcp")

# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------

_client: Optional[GeminiWebClient] = None
_browser: str = "chrome"
_profile: str = "Default"
_language: str = "zh-CN"
_cookie_file: Optional[str] = None
_metadata_cache: dict[str, list[Any]] = {}  # conversation_id -> metadata

# Hot-reload baseline (epoch seconds): the cookie-file mtime the current
# _client already reflects. Advanced after every client build AND after our
# own _persist_cookies write (rotation rewrites the file, which must never
# look like an external replacement — no read-your-own-write loop).
_cookies_baseline_at: float = 0.0

# How often to rotate __Secure-1PSIDTS in the background (seconds).
_ROTATE_INTERVAL = 25 * 60

# ---------------------------------------------------------------------------
# Conversation session table (fingerprint chain for stateless clients)
#
# OpenAI clients (DSH) replay the full message history every turn and never
# pass conversation ids. We fingerprint "the history BEFORE the last user
# turn" (parent key) and "the history AFTER our reply" (child key): the next
# request's parent key equals this turn's child key, which chains turns onto
# one web-side conversation. A miss → fresh conversation with the full
# history rendered as a transcript (semantically lossless, one extra turn).
# ---------------------------------------------------------------------------

_CONV_TTL = 90 * 60     # web-side conversations outlive several rotations
_CONV_MAX = 200         # single-user daemon: plenty


@dataclass
class _ConvRecord:
    metadata: list[Any]
    model: Optional[str]
    created_at: float
    updated_at: float


_conv_by_cid: dict[str, "_ConvRecord"] = {}
_conv_by_key: dict[str, str] = {}  # fingerprint -> conversation_id

# Guards the compound read/modify/write sequences on the session table
# (register writes cid + two keys + evicts; lookup reads key -> cid -> record
# across two dicts). RLock because conv_register calls into _conv_evict.
_CONV_LOCK = threading.RLock()


def session_key(texts: list[str], model: str) -> str:
    """Fingerprint an ordered history of flattened texts under one model."""
    h = hashlib.sha256()
    h.update(model.encode("utf-8", "ignore"))
    for t in texts:
        h.update(b"\x1e")
        h.update(t.strip().encode("utf-8", "ignore"))
    return h.hexdigest()


def _conv_evict() -> None:
    """TTL + max-entries eviction; fingerprint keys cleaned with records.

    Caller must hold ``_CONV_LOCK``. Orphan fingerprint keys (pointing at a
    conversation id that no longer has a record) are removed UNCONDITIONALLY:
    TTL and capacity trimming above are not the only ways a cid can vanish,
    and a stale key would chain a future request onto a dead conversation.
    """
    now = time.time()
    stale = [cid for cid, r in _conv_by_cid.items() if now - r.updated_at > _CONV_TTL]
    for cid in stale:
        _conv_by_cid.pop(cid, None)
    if len(_conv_by_cid) > _CONV_MAX:
        for cid, _ in sorted(_conv_by_cid.items(), key=lambda kv: kv[1].updated_at)[
            : len(_conv_by_cid) - _CONV_MAX
        ]:
            _conv_by_cid.pop(cid, None)
    for k in [k for k, cid in _conv_by_key.items() if cid not in _conv_by_cid]:
        _conv_by_key.pop(k, None)


def conv_register(result: Any, texts_after: list[str], model: str) -> str:
    """Register a completed turn: cid record + parent/child fingerprint keys.

    ``texts_after`` is the flattened history INCLUDING our reply — its
    prefix minus the trailing assistant entry is this turn's parent key and
    the whole list is the child key the NEXT request will fingerprint.
    Re-registering the same turn (e.g. a client retrying the identical
    request) simply overwrites the same keys — lookups stay stable.
    """
    cid = result.conversation_id or ""
    now = time.time()
    with _CONV_LOCK:
        if cid:
            _conv_by_cid[cid] = _ConvRecord(
                metadata=result.metadata or [], model=model, created_at=now, updated_at=now
            )
        if len(texts_after) >= 1:
            _conv_by_key[session_key(texts_after[:-1], model)] = cid
        _conv_by_key[session_key(texts_after, model)] = cid
        _conv_evict()
    return cid


def conv_lookup(texts: list[str], model: str, conv_id: Optional[str] = None) -> Optional["_ConvRecord"]:
    """Find a live record by explicit id, else by the parent fingerprint."""
    with _CONV_LOCK:
        if conv_id:
            rec = _conv_by_cid.get(conv_id)
            if rec and time.time() - rec.updated_at <= _CONV_TTL:
                return rec
            return None
        if not texts:
            return None
        cid = _conv_by_key.get(session_key(texts, model))
        if not cid:
            return None
        rec = _conv_by_cid.get(cid)
        if rec and time.time() - rec.updated_at <= _CONV_TTL:
            return rec
        return None

# Single-cookie single-writer: the web session must not be driven
# concurrently (review-20261102 P1-1). stdio was naturally serial; both the
# HTTP MCP tools, the REST layer and streaming workers share ONE lock
# instance owned by the client module (also covering account RPCs and the
# rotation thread).
_GENERATE_LOCK = GENERATE_LOCK


class _TokenAuthMiddleware:
    """ASGI middleware requiring ``Authorization: Bearer <token>`` on HTTP.

    Only HTTP requests are gated; lifespan and websocket scopes pass through
    untouched. ``token=None`` disables the check (opt-in via
    ``--http-allow-anonymous``). The comparison is constant-time.
    """

    def __init__(self, app: Any, token: Optional[str]) -> None:
        self.app = app
        self._expected = b"Bearer " + token.encode() if token else None

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") == "http" and self._expected is not None:
            headers = dict(scope.get("headers") or [])
            if not hmac.compare_digest(
                headers.get(b"authorization", b""), self._expected
            ):
                await self._deny(send)
                return
        await self.app(scope, receive, send)

    @staticmethod
    async def _deny(send: Any) -> None:
        body = b'{"error": "invalid_token", "error_description": "Authentication required"}'
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"www-authenticate", b'Bearer error="invalid_token"'),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


async def _run_http_async(host: str, port: int, token: Optional[str]) -> None:
    """Run the streamable-http transport plus the REST facade on one port.

    A thin ASGI dispatcher splits by path: ``/mcp*`` keeps serving the MCP
    streamable app behind the bearer middleware (lifespan included, so the
    MCP session manager still starts); everything else goes to the
    OpenAI-compatible REST layer in :mod:`gemini_mcp.api`, which enforces the
    same token itself and exempts ``/healthz`` for control-plane probes.
    """
    import uvicorn

    from .api import build_api_app

    mcp_asgi = _TokenAuthMiddleware(mcp.streamable_http_app(host=host), token)
    app = _DispatchApp(mcp_asgi, build_api_app(token))
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    await uvicorn.Server(config).serve()


class _DispatchApp:
    """Route ASGI http scopes by path; hand every other scope to the MCP app.

    The lifespan scope (uvicorn startup/shutdown) must reach the MCP
    streamable app — its session manager starts there — while the REST app
    has no lifespan requirements and tolerates never seeing one.
    """

    def __init__(self, mcp_asgi: Any, api_asgi: Any) -> None:
        self._mcp = mcp_asgi
        self._api = api_asgi

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self._mcp(scope, receive, send)
            return
        path = scope.get("path", "")
        if path == "/mcp" or path.startswith("/mcp/"):
            await self._mcp(scope, receive, send)
        else:
            await self._api(scope, receive, send)


# Well-known fallback cookie location: marketplace users just drop their
# exported cookies.json here — no env vars or patch edits required.
DEFAULT_COOKIE_FILE = Path.home() / ".config" / "gemini-web-mcp" / "cookies.json"

# Whether the current session's cookies were loaded from a file (vs. browser
# auto-extraction). _persist_cookies only writes when this is true or when an
# explicit file path was configured — an auto-extract session must never
# silently create a plaintext credential file.
_cookies_from_file = False


def _cookie_file_path() -> Optional[str]:
    """Resolve the cookie file: CLI arg > GEMINI_COOKIE_FILE > default path."""
    path = _cookie_file or os.environ.get("GEMINI_COOKIE_FILE")
    if path:
        return path
    return str(DEFAULT_COOKIE_FILE) if DEFAULT_COOKIE_FILE.is_file() else None


def _load_cookies() -> tuple[dict[str, str], list[str]]:
    """Load cookies: explicit file (CLI / env / default path) first, then
    auto-extract from the local browser profile."""
    global _cookies_from_file
    _cookies_from_file = False
    path = _cookie_file_path()
    if path and Path(path).is_file():
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            return {}, [f"cookie file is not readable JSON: {path} ({e})"]
        if isinstance(raw, dict):
            _cookies_from_file = True
            return raw, []
        if isinstance(raw, list):
            _cookies_from_file = True
            return {c["name"]: c["value"] for c in raw if c.get("value")}, []
    return cookies_for_gemini(_browser, _profile)


def _advance_cookie_baseline() -> None:
    """Move the hot-reload baseline up to "what the client already reflects".

    Called after a client build (the jar was just read from the file) and
    after a successful persist (our own write bumped the mtime). The file's
    CURRENT mtime becomes the baseline, so only a strictly newer external
    replacement triggers a reload.
    """
    global _cookies_baseline_at
    path = _cookie_file_path()
    try:
        mtime = os.path.getmtime(path) if path and os.path.isfile(path) else time.time()
    except OSError:
        mtime = time.time()
    _cookies_baseline_at = max(_cookies_baseline_at, mtime)


def _get_client() -> GeminiWebClient:
    """Process-level client singleton (double-checked locking).

    Lock-free fast path once built. On a None singleton (cold start, or the
    instant _maybe_reload_client clears it for a rebuild) the build happens
    INSIDE _GENERATE_LOCK after a second check — without it, two lock-free
    callers could both see None and each construct a client, breaking the
    singleton (two live sessions driving one account). _maybe_reload_client
    takes the same single lock (re-entrant), so no lock-order deadlock face.
    """
    global _client
    if _client is not None:
        return _client
    with _GENERATE_LOCK:
        if _client is not None:  # double-check: a concurrent builder won
            return _client
        cookies, errors = _load_cookies()
        if not cookies:
            raise GeminiWebError(
                "No Google cookies found. " + ("; ".join(errors) if errors else "")
            )
        _client = GeminiWebClient(cookies, language=_language)
        _advance_cookie_baseline()
        return _client


def _maybe_reload_client() -> bool:
    """Hot-reload the client singleton when the cookie file changed on disk.

    This daemon is the single writer of rotated cookies, but an external dump
    tool may replace the whole cookie file at any time — the process-level
    singleton only reads it at construction. Both hot paths (every generate
    via _generate_serialized, and the _status_payload probe used by recovery
    orchestrators) call this first: stat the file's mtime and rebuild the
    singleton under the generate lock when it is strictly newer than
    _cookies_baseline_at (our own persist advances the baseline, so
    self-writes never trigger a reload).

    Returns True when the client was rebuilt. A broken new file (bad JSON,
    missing key cookies) keeps the OLD client and returns False — the process
    must not die; the next hot path retries once the file is fixed.
    """
    global _client
    if _client is None:
        return False  # nothing to reload; lazy init in _get_client sets the baseline
    path = _cookie_file_path()
    if not path:
        return False  # browser-extraction session: no file to watch
    try:
        if os.path.getmtime(path) <= _cookies_baseline_at:
            return False
    except OSError:
        return False
    with _GENERATE_LOCK:
        # Re-check under the lock: a concurrent reload may have won the race
        # and already advanced the baseline.
        try:
            if os.path.getmtime(path) <= _cookies_baseline_at:
                return False
        except OSError:
            return False
        old = _client
        _client = None  # _get_client() re-reads via _load_cookies()
        try:
            _get_client()
        except Exception as e:  # noqa: BLE001 - keep the old client alive
            _client = old
            log.warning(
                "Cookie hot-reload failed (new file unusable), keeping previous "
                "client: %s",
                e,
            )
            return False
    log.info("Cookie file replaced externally; client hot-reloaded from %s", path)
    return True


def _persist_cookies(client: GeminiWebClient) -> None:
    """Write the (rotated) session cookies back to the cookie file, so a
    process restart keeps using fresh cookies. Skips sessions that loaded
    cookies purely from browser auto-extraction (no explicit file), so a
    plaintext credential file is never created without the user's choice."""
    if not (_cookies_from_file or _cookie_file or os.environ.get("GEMINI_COOKIE_FILE")):
        return
    path = _cookie_file_path()
    if not path:
        return
    entries = []
    for cookie in client.session.cookies:
        if "google.com" in (cookie.domain or ""):
            entries.append(
                {
                    "name": cookie.name,
                    "value": cookie.value,
                    "domain": cookie.domain,
                    "path": cookie.path or "/",
                }
            )
    if entries:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(
                json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.chmod(path, 0o600)
            # Our own write bumped the mtime — advance the reload baseline so
            # the next hot-path stat does not mistake it for an external
            # replacement (read-your-own-write loop).
            _advance_cookie_baseline()
            log.info("Rotated cookies persisted to %s (%d cookies)", path, len(entries))
        except OSError as e:
            log.warning("Failed to persist rotated cookies: %s", e)


def _prewarm_account_rpcs() -> None:
    """Populate the model-discovery cache once, in the background.

    Retries with backoff: cookies may still be settling right after startup.
    Failures are silent by design — the static catalog keeps serving and the
    snapshot stays None until a later attempt succeeds."""
    import time as _time

    _time.sleep(5)  # let the cookie file load and /healthz go ready first
    for attempt in range(3):
        try:
            client = _get_client()
            with _GENERATE_LOCK:
                registry = client.discover_models()
            log.info(
                "Model discovery prewarmed: %d models, capacity %d",
                len(registry.get("models", [])),
                registry.get("capacity", 0),
            )
            return
        except Exception as e:  # noqa: BLE001 - background best-effort
            log.warning("Model discovery prewarm attempt %d failed: %s", attempt + 1, e)
            _time.sleep(60 * (attempt + 1))


def _rotate_once() -> None:
    """One rotation cycle (extracted from _auto_rotate_loop for testability).

    ``skip_if_fresh`` + the resolved persist path keep read/write on one
    file, so concurrent gemini-web clients never turn the shared
    RotateCookies endpoint into a 429 fight. Rotation and persist run under
    the generate lock: requests.Session is not thread-safe and the jar merge
    must not race a streaming response.

    Observability (state.py): rotated=False is a silent auth break — log a
    WARNING (the package logger feeds the sanitized event log) and bump
    auth_failures; a success clears the counter. The counters surface on
    /v1/status (cookie_rotation.last_result) and /healthz (auth_failures).

    The singleton reference is taken INSIDE the lock: grabbing it earlier and
    then waiting on the lock means a hot-reload that rebuilds the singleton
    while we wait would leave us rotating — and persisting! — the STALE jar,
    overwriting an externally dumped cookie file with old cookies (a false
    success).
    """
    try:
        with _GENERATE_LOCK:
            client = _get_client()  # latest singleton, under the lock
            rotated = client.rotate_cookies(
                force=True, skip_if_fresh=True, cookie_file=_cookie_file_path()
            )
            if rotated:
                _persist_cookies(client)
        if rotated:
            state.record_rotation(ok=True)
        else:
            failures = state.record_rotation(ok=False)
            log.warning(
                "Background cookie rotation FAILED (auth_failures=%d): no fresh "
                "__Secure-1PSIDTS obtained — cookies may be rejected/stale",
                failures,
            )
    except Exception:  # noqa: BLE001 - keep the loop alive
        log.warning("Background cookie rotation failed", exc_info=True)


def _auto_rotate_loop() -> None:
    """Background daemon: renew the short-lived 1PSIDTS cookie periodically.
    Sleep carries a small jitter so a fleet of clients does not sync its
    rotation bursts."""
    while True:
        time.sleep(_ROTATE_INTERVAL + random.uniform(-15, 15))
        _rotate_once()


def _metadata_for(conversation_id: Optional[str], response_id: Optional[str]) -> Optional[list[Any]]:
    if not conversation_id:
        return None
    meta = list(_metadata_cache.get(conversation_id, DEFAULT_METADATA.copy()))
    meta[0] = conversation_id
    if response_id:
        meta[1] = response_id
    return meta


def _remember(result) -> None:
    if result.conversation_id and result.metadata:
        _metadata_cache[result.conversation_id] = result.metadata


mcp = MCPServer(
    "gemini-web",
    # serverInfo.version for the MCP initialize handshake (was an empty
    # string before); sourced from the installed distribution metadata.
    version=__version__,
    instructions=(
        "Bridges to gemini.google.com using the logged-in browser session "
        "(cookie-based auth, no API key). Use gemini_status to check auth, "
        "gemini_send_message to chat. To continue a conversation, pass the "
        "conversation_id (and optionally response_id) returned by the "
        "previous call."
    ),
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

# Required session cookies (full list Google may expect; only __Secure-1PSID
# is mandatory, __Secure-1PSIDTS is mandatory if present in your browser).
REQUIRED_COOKIES = [
    "__Secure-1PSID",
    "__Secure-1PSIDTS",
    "SID",
    "HSID",
    "SSID",
    "APISID",
    "SAPISID",
    "__Secure-1PAPISID",
    "__Secure-3PSID",
    "__Secure-3PSIDTS",
    "__Secure-3PAPISID",
    "NID",
    "AEC",
    "SIDCC",
    "__Secure-1PSIDCC",
    "1P_JAR",
]


def _status_payload() -> dict[str, Any]:
    """Connection diagnostics shared by the gemini_status tool and GET /v1/status."""
    import platform

    # Status is the recovery orchestrator's probe: run the hot-reload check
    # FIRST so an externally replaced cookie file is visible to the client
    # diagnostics below (not just to the raw file read).
    _maybe_reload_client()
    cookies, errors = _load_cookies()
    key_names = [n for n in REQUIRED_COOKIES if n in cookies]
    missing = [n for n in REQUIRED_COOKIES if n not in cookies]
    auth_state = state.snapshot()
    status: dict[str, Any] = {
        "ok": False,
        "browser": _browser,
        "profile": _profile,
        "platform": platform.system(),
        "cookie_count": len(cookies),
        "cookies_present": key_names,
        "cookies_missing": missing,
        "errors": errors,
        # Rotation-failure observability (state.py): consecutive failed
        # background rotations; 0 while healthy. Present in EVERY branch —
        # the recovery orchestrator probes exactly when things are broken.
        "auth_failures": auth_state["auth_failures"],
        "cookie_rotation": {"last_result": auth_state["last_result"]},
    }
    if not cookies:
        status["errors"] = errors or ["no cookies available"]
        return status
    if "__Secure-1PSID" not in cookies:
        status["errors"] = [
            "The mandatory '__Secure-1PSID' session cookie is missing. "
            "Log in to gemini.google.com and export the cookies (Cookie-Editor "
            "extension), then pass them via --cookie-file."
        ]
        return status
    try:
        client = _get_client()
        with _GENERATE_LOCK:  # init() drives the shared session (real GET)
            client.init(force=True)
        status["ok"] = True
        status["token_fetched"] = True
        status["language"] = client.language
        status["model"] = client.model_description
        status["session_id_present"] = bool(client.session_id)
        status["cookie_rotation"].update(
            {
                "last_rotate_seconds_ago": (
                    round(time.time() - client._last_rotate_at)
                    if client._last_rotate_at else None
                ),
                "interval_seconds": _ROTATE_INTERVAL,
                "persist_to": _cookie_file_path() if _cookies_from_file else None,
            }
        )
    except Exception as e:  # noqa: BLE001 - diagnostics tool must not crash
        status["errors"] = [str(e)]
    return status


@mcp.tool()
def gemini_status() -> dict[str, Any]:
    """Diagnose the connection to gemini.google.com: cookie availability, token freshness and reachability."""
    return _status_payload()


@mcp.tool()
def gemini_models() -> dict[str, Any]:
    """List selectable model tiers and thinking levels.

    Use the returned names with the ``model`` / ``thinking`` arguments of
    ``gemini_send_message`` (and the ``model`` field / ``gemini_thinking``
    body field of the OpenAI-compatible ``/v1/chat/completions`` endpoint).
    """
    tiers = []
    for name, (model_id, capacity, number) in MODEL_CATALOG.items():
        tiers.append(
            {
                "name": name,
                "openai_id": f"gemini-{name}",
                "capacity": capacity,
                "mode_number": number,
            }
        )
    return {
        "default_model": DEFAULT_MODEL,
        "default_thinking": DEFAULT_THINKING,
        "fallback_chain": list(DEFAULT_MODEL_FALLBACK_CHAIN),
        "tiers": tiers,
        "model_aliases": MODEL_ALIASES,
        "thinking_levels": THINKING_LEVELS,
        "thinking_notes": (
            "1=standard, 2=extended (default, 低=1/中=2/高=2); 3=Deep Think is an "
            "async placeholder and is rejected by the REST facade "
            "(deep_think_unavailable)"
        ),
    }


def _usage_payload() -> dict[str, Any]:
    """Local usage counters + server-side plan truth, shared by tool and REST.

    Two sources: (1) process-local accounting — request counts, character
    estimates, model-fallback events; (2) the web backend's own usage-limits
    data via account RPCs (qpEbW request-quota buckets + jSf9Qc compute-based
    5h/weekly windows with AI credits). Either server source failing degrades
    gracefully; ``server_quota.alert_level`` rolls the worst bucket into
    ok/warning/critical/exhausted for clients.
    """
    from .usage import snapshot

    data: dict[str, Any] = snapshot()
    try:
        client = _get_client()
        data["model"] = client.model_description
        data["fallback_events"] = list(client.fallback_events)[-20:]
        data["cookie_rotation"] = {
            "last_rotate_seconds_ago": (
                round(time.time() - client._last_rotate_at)
                if client._last_rotate_at else None
            ),
            "interval_seconds": _ROTATE_INTERVAL,
            "persist_to": _cookie_file_path() if _cookies_from_file else None,
        }
        server: dict[str, Any] = {}
        # Account RPCs drive the same requests.Session as generate(). Try to
        # take the generate lock without blocking: a hung/slow generate must
        # NOT stall usage queries (observed: 45s+ block during an upstream
        # stall). Busy → degrade to local-only accounting immediately.
        got_lock = _GENERATE_LOCK.acquire(blocking=False)
        if not got_lock:
            server["ok"] = False
            server["reason"] = "generate in progress; server quota skipped this call"
        else:
            try:
                try:
                    server["quotas"] = client.fetch_quota()
                except Exception as e:  # noqa: BLE001 - per-source degradation
                    server["quotas_error"] = str(e)
                try:
                    server["usage_info"] = client.fetch_usage_info()
                except Exception as e:  # noqa: BLE001 - per-source degradation
                    server["usage_info_error"] = str(e)
            finally:
                _GENERATE_LOCK.release()
        server["ok"] = bool(server.get("quotas") or server.get("usage_info"))
        levels: list[float] = []
        for bucket in server.get("quotas", {}).values():
            pct = bucket.get("usage_percentage")
            if isinstance(pct, (int, float)):
                levels.append(float(pct))
        usage_info = server.get("usage_info") or {}
        for key in ("current_5h", "weekly"):
            window = usage_info.get(key) or {}
            pct = window.get("usage_percentage") if isinstance(window, dict) else None
            if isinstance(pct, (int, float)):
                levels.append(float(pct))
        worst = max(levels) if levels else None
        server["worst_usage_percentage"] = worst
        if not server["ok"] and worst is None:
            server["alert_level"] = "unknown"  # both sources failed — no truth
        else:
            server["alert_level"] = (
                "exhausted" if worst is not None and worst >= 100
                else "critical" if worst is not None and worst >= 90
                else "warning" if worst is not None and worst >= 75
                else "ok"
            )
        data["server_quota"] = server
    except Exception as e:  # noqa: BLE001 - diagnostics tool must not crash
        data["errors"] = [str(e)]
    return data


@mcp.tool()
def gemini_usage() -> dict[str, Any]:
    """Local usage counters (requests, size estimates, model fallback events) + session health.

    Google's web backend exposes no per-account quota API for the Gemini web
    plan; this reports what this process has driven, plus degradation events
    that indicate plan limits being hit upstream.
    """
    return _usage_payload()


def _generate_serialized(
    prompt: str,
    metadata: Optional[list[Any]] = None,
    files: Optional[list[str]] = None,
    model: Optional[str] = None,
    thinking: Optional[Union[int, str]] = None,
):
    """Run one generate call under the process-wide lock and remember the turn.

    Usage accounting is left to the callers (the MCP tool and the REST layer
    each count their own traffic exactly once). Entry hot-reload check: an
    externally replaced cookie file is picked up here (stat cost only).
    """
    _maybe_reload_client()
    client = _get_client()
    with _GENERATE_LOCK:
        result = client.generate(
            prompt, metadata=metadata, files=files, model=model, thinking=thinking
        )
    _remember(result)
    return result


@mcp.tool()
def gemini_send_message(
    message: str,
    conversation_id: Optional[str] = None,
    response_id: Optional[str] = None,
    language: Optional[str] = None,
    file_paths: Optional[list[str]] = None,
    model: Optional[str] = None,
    thinking: Optional[Union[int, str]] = None,
) -> dict[str, Any]:
    """Send a message to Gemini (web interface), optionally attaching files.

    Without conversation_id a brand-new chat is started. Pass the
    conversation_id (and optionally response_id) from a previous result to
    continue that conversation.

    Args:
        message: The prompt to send.
        conversation_id: Continue this conversation (from a previous call).
        response_id: The response id to reply under (from a previous call).
        language: UI/reply language hint, e.g. "zh-CN" (default), "en".
        file_paths: Local paths of images/videos to attach (uploaded to
            Gemini and analyzed together with the message).
        model: Per-call model tier override (e.g. "argon", "pro", "flash",
            "flash-lite", "legacy"). See gemini_models for the full list.
            Unset keeps the process default (GEMINI_MODEL, default argon).
        thinking: Per-call thinking depth override (1=standard, 2=extended —
            低=1/中=2/高=2; level 3 Deep Think is an async placeholder and
            raises DeepThinkUnavailable). Unset keeps the process default
            (GEMINI_THINKING, default 2).

    Returns:
        The model's answer plus conversation_id/response_id/candidate_id to
        continue the chat later, and any web images cited in the reply. The
        returned ``model`` field is the model name the web server says served
        the answer (e.g. "4 Argon High").

    Model choice:
        ``GEMINI_MODEL`` (default ``argon`` = Gemini 4 "Argon", the strongest
        model the web UI exposes) and ``GEMINI_THINKING`` (default ``2`` =
        extended; ``3`` = Deep Think, which answers asynchronously) configure
        the tier. Rejections degrade Argon -> 3.1 Pro -> Flash-Basic -> the
        model-less legacy request. ``GEMINI_MODEL=legacy`` restores the
        pre-model-selection behaviour.
    """
    if not message or not message.strip():
        raise ValueError("message must not be empty")
    client = _get_client()
    if language and language != client.language:
        client.language = language
    meta = _metadata_for(conversation_id, response_id)
    try:
        result = _generate_serialized(
            message.strip(), metadata=meta, files=file_paths,
            model=model, thinking=thinking,
        )
    except Exception:  # noqa: BLE001 - count failed turns for the usage surface
        from . import usage

        usage.record_error()
        raise
    from . import usage

    usage.record_request(chars_in=len(message), chars_out=len(result.text))
    return result.to_dict()


MAX_REVIEW_FILES = 20
MAX_REVIEW_BYTES = 2_000_000


def _resolve_review_paths(patterns: list[str]) -> list[str]:
    """Expand globs / env vars into a de-duplicated list of real files."""
    resolved: list[str] = []
    for pattern in patterns:
        expanded = os.path.expandvars(os.path.expanduser(pattern))
        matches = sorted(glob.glob(expanded, recursive=True))
        candidates = matches if matches else [expanded]
        for candidate in candidates:
            if os.path.isfile(candidate):
                resolved.append(os.path.abspath(candidate))
    seen: set[str] = set()
    unique: list[str] = []
    for path in resolved:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


@mcp.tool()
def gemini_review(
    paths: list[str],
    question: str,
    conversation_id: Optional[str] = None,
    response_id: Optional[str] = None,
) -> dict[str, Any]:
    """Upload local source files to Gemini and ask it to review them.

    Practical alternative to Gemini's GitHub import on the web interface: that
    import hands the model a compressed binary blob it cannot read, whereas
    uploading the files themselves works. Globs are expanded, so
    ``["src/**/*.py"]`` is valid.

    Args:
        paths: Files or glob patterns to upload (each <= 2 MB, at most 20 files).
        question: What to ask about them, e.g. "review these for bugs".
        conversation_id: Continue an earlier review (from a previous result).
        response_id: Response id to reply under, if continuing.

    Returns:
        The answer plus ``model`` (the tier that actually served it) and
        ``files`` / ``skipped`` describing what was uploaded.
    """
    if not question or not question.strip():
        raise ValueError("question must not be empty")
    if not paths:
        raise ValueError("paths must not be empty")

    selected: list[str] = []
    skipped: list[dict[str, Any]] = []
    for path in _resolve_review_paths(paths):
        try:
            size = os.path.getsize(path)
        except OSError as error:
            skipped.append({"path": path, "reason": str(error)})
            continue
        if size > MAX_REVIEW_BYTES:
            skipped.append({"path": path, "reason": "larger than %d bytes" % MAX_REVIEW_BYTES})
            continue
        if len(selected) >= MAX_REVIEW_FILES:
            skipped.append({"path": path, "reason": "more than %d files" % MAX_REVIEW_FILES})
            continue
        selected.append(path)
    if not selected:
        raise ValueError("no uploadable files matched")

    client = _get_client()
    meta = _metadata_for(conversation_id, response_id)
    started = time.time()
    result = _generate_serialized(question.strip(), metadata=meta, files=selected)
    payload = result.to_dict()
    payload["files"] = [{"path": p, "bytes": os.path.getsize(p)} for p in selected]
    payload["skipped"] = skipped
    payload["elapsed_seconds"] = round(time.time() - started, 2)
    return payload

@mcp.tool()
def gemini_analyze_media(
    file_path: str,
    prompt: str = (
        "请客观描述这张图片/这个视频的可见内容：画面主体与物品、场景与背景、"
        "构图、光线与色彩风格、出现的文字或标识。请直接陈述视觉事实，"
        "按条目分点说明。"
    ),
    conversation_id: Optional[str] = None,
    response_id: Optional[str] = None,
) -> dict[str, Any]:
    """Use Gemini's eyes: upload one image/video and get a description.

    This is the tool an agent should call when the user uploads an image or
    video — Gemini looks at it and the result text comes back here.

    Tip: the more specific the prompt, the more likely Gemini answers — if a
    broad request is refused, retry with an objective angle (composition,
    lighting, colors, objects, text).

    Args:
        file_path: Local path of the image or video file.
        prompt: What to ask about the media (default: objective visual facts).
        conversation_id: Optional conversation to continue within.

    Returns:
        {text, media (cited web images), conversation_id, response_id}.
    """
    if not file_path:
        raise ValueError("file_path must not be empty")
    client = _get_client()
    meta = _metadata_for(conversation_id, response_id)
    result = _generate_serialized(prompt, metadata=meta, files=[file_path])
    _remember(result)
    return result.to_dict()


@mcp.tool()
def gemini_generate_image(
    prompt: str,
    reference_image: Optional[str] = None,
    save_dir: str = "./media",
    conversation_id: Optional[str] = None,
) -> dict[str, Any]:
    """Ask Gemini to generate an image (Imagen) from a text prompt.

    Optionally pass a reference_image (local path) for image-to-image edits.

    Args:
        prompt: What image to generate, e.g. "一只在月球上打伞的猫，水彩风格".
        reference_image: Optional local image path to edit / use as reference.
        save_dir: Directory the generated image is downloaded to.
        conversation_id: Optional existing conversation to generate within.

    Returns:
        {text, generated images (local paths + URLs), conversation_id}.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")
    client = _get_client()
    files = [reference_image] if reference_image else None
    meta = _metadata_for(conversation_id, None)
    result = _generate_serialized(prompt.strip(), metadata=meta, files=files)
    _remember(result)
    return _with_saved_media(result, save_dir, kinds={"generated_image"})


@mcp.tool()
def gemini_generate_video(
    prompt: str,
    save_dir: str = "./media",
    conversation_id: Optional[str] = None,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Ask Gemini to generate a video (Veo) from a text prompt.

    Video generation is asynchronous: this tool sends the request, then polls
    the conversation until Gemini finishes rendering and returns the video URL.

    Args:
        prompt: What video to generate, e.g. "一只橘猫在窗台上看下雨，电影感".
        save_dir: Directory the generated video is downloaded to.
        conversation_id: Optional existing conversation to generate within.
        timeout_seconds: Max time to wait for rendering (default 600s).

    Returns:
        {text, generated video (local path + URL), conversation_id}.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")
    client = _get_client()
    meta = _metadata_for(conversation_id, None)
    result = _generate_serialized(prompt.strip(), metadata=meta)
    _remember(result)
    videos = [m for m in result.media if m.kind == "generated_video"]

    # Poll read_conversation while the video is still rendering.
    cid = result.conversation_id
    deadline = time.monotonic() + timeout_seconds
    while not videos and cid and time.monotonic() < deadline:
        time.sleep(15)
        try:
            history = client.read_conversation(cid, limit=3)
            for turn in history.get("turns", []):
                for item in turn.get("media", []) or []:
                    if item["kind"] == "generated_video" and item.get("url"):
                        videos.append(
                            MediaRef(kind="generated_video", url=item["url"])
                        )
        except Exception:  # noqa: BLE001 - keep polling on transient errors
            continue
    if not videos:
        return {**result.to_dict(), "generated_video": None,
                "note": "Gemini did not return a video yet within the timeout. "
                        "Check the conversation later with gemini_read_conversation."}
    result.media = [m for m in result.media if m.kind != "generated_video"] + videos
    return _with_saved_media(result, save_dir, kinds={"generated_video"})


@mcp.tool()
def gemini_download_media(
    url: str,
    save_path: str,
) -> dict[str, str]:
    """Download a media URL (googleusercontent.com) to a local file.

    Some media URLs only work with the logged-in session cookies; this tool
    downloads them through the authenticated session.

    Args:
        url: The media URL (e.g. from a generated image result).
        save_path: Where to save the file (a .png/.mp4 suffix is added if missing).

    Returns:
        {path: local file path}
    """
    if not url or not save_path:
        raise ValueError("url and save_path must not be empty")
    with _GENERATE_LOCK:  # shared session: single-writer with generate
        path = _get_client().download_media(url, save_path)
    return {"path": path}


def _with_saved_media(
    result: Any, save_dir: str, kinds: set[str]
) -> dict[str, Any]:
    """Download generated media into save_dir and return result dict with local paths."""
    import os

    out = result.to_dict()
    saved = []
    for i, m in enumerate(out.get("media", [])):
        if m["kind"] not in kinds:
            continue
        ext = ".mp4" if m["kind"] == "generated_video" else ".png"
        dest = os.path.join(save_dir, f"{m['kind']}_{result.conversation_id or 'new'}_{i}{ext}")
        try:
            with _GENERATE_LOCK:
                local = _get_client().download_media(m["url"], dest)
            m["local_path"] = local
            saved.append(m)
        except Exception as e:  # noqa: BLE001 - report download failure, keep URL
            m["download_error"] = str(e)
            saved.append(m)
    out["saved_media"] = saved
    return out


@mcp.tool()
def gemini_list_conversations(limit: int = 13) -> list[dict[str, Any]]:
    """List the most recent conversations of the logged-in account.

    Args:
        limit: How many conversations per bucket (pinned/unpinned) to fetch.

    Returns:
        A list of {conversation_id, title, pinned, timestamp}.
    """
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    with _GENERATE_LOCK:  # shared session: single-writer with generate
        return _get_client().list_conversations(limit=limit)


@mcp.tool()
def gemini_read_conversation(conversation_id: str, limit: int = 10) -> dict[str, Any]:
    """Read the turns of an existing conversation (newest first).

    Args:
        conversation_id: e.g. "c_..." as returned by gemini_list_conversations.
        limit: Maximum number of turns to fetch.

    Returns:
        {conversation_id, turns: [{role, text, ...}]}
    """
    if not conversation_id:
        raise ValueError("conversation_id must not be empty")
    with _GENERATE_LOCK:
        return _get_client().read_conversation(conversation_id, limit=limit)


@mcp.tool()
def gemini_delete_conversation(conversation_id: str) -> dict[str, str]:
    """Delete a conversation from Gemini history (irreversible).

    Args:
        conversation_id: e.g. "c_..." as returned by gemini_list_conversations.
    """
    if not conversation_id:
        raise ValueError("conversation_id must not be empty")
    with _GENERATE_LOCK:
        _get_client().delete_conversation(conversation_id)
    _metadata_cache.pop(conversation_id, None)
    return {"deleted": conversation_id}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    global _browser, _profile, _language, _cookie_file

    parser = argparse.ArgumentParser(
        prog="gemini-mcp",
        description="MCP server bridging agents to gemini.google.com via the web interface (cookie auth).",
    )
    parser.add_argument("--browser", choices=["chrome", "edge", "chromium"], default="chrome",
                        help="Browser whose profile holds the Google session (default: chrome).")
    parser.add_argument("--profile", default="Default",
                        help="Browser profile directory (default: Default).")
    parser.add_argument("--language", default="zh-CN",
                        help="Language hint sent to Gemini (default: zh-CN).")
    parser.add_argument("--cookie-file", default=None,
                        help="JSON file with cookies [{name,value,...}] or {name:value}; overrides browser extraction.")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio",
                        help="MCP transport (default: stdio).")
    parser.add_argument("--port", type=int, default=8900,
                        help="HTTP port when --transport http (default: 8900).")
    parser.add_argument("--http-token", default=None,
                        help="Bearer token required for --transport http. Defaults to "
                             "GEMINI_HTTP_TOKEN, then to a randomly generated token "
                             "printed at startup.")
    parser.add_argument("--http-allow-anonymous", action="store_true",
                        help="Disable HTTP token auth. DANGEROUS: any local process "
                             "(and any website you visit) can drive the logged-in "
                             "Google account through the HTTP port.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Sanitized daemon event log (rotating file, metadata only — see
    # eventlog.py). Best effort: a broken log path must not kill the daemon.
    from .eventlog import configure_event_logging

    try:
        event_path = configure_event_logging()
    except OSError as e:
        event_path = None
        log.warning("event log unavailable: %s", e)
    if event_path is not None:
        log.info("event log configured at %s", event_path)
    if args.cookie_file:
        os.environ["GEMINI_COOKIE_FILE"] = args.cookie_file
    _cookie_file = args.cookie_file or os.environ.get("GEMINI_COOKIE_FILE")
    _browser, _profile, _language = args.browser, args.profile, args.language

    # Background daemon keeps the short-lived __Secure-1PSIDTS fresh by
    # rotating it through accounts.google.com/RotateCookies every 25 minutes
    # and persisting the new cookies back to the cookie file.
    import threading

    threading.Thread(target=_auto_rotate_loop, daemon=True, name="gemini-cookie-rotator").start()
    # Warm the model-discovery cache in the background so /v1/models (and any
    # consumer of the roster) is served instantly from cache and never hangs
    # on the network or the generate lock.
    threading.Thread(target=_prewarm_account_rpcs, daemon=True, name="gemini-prewarm").start()

    if args.transport == "http":
        import asyncio
        import secrets

        token: Optional[str] = args.http_token or os.environ.get("GEMINI_HTTP_TOKEN")
        if args.http_allow_anonymous:
            token = None
            log.warning(
                "HTTP token auth disabled (--http-allow-anonymous): any local "
                "process can drive this logged-in Google account."
            )
        elif not token:
            token = secrets.token_urlsafe(24)
            print(
                f"gemini-web-mcp HTTP auth token (send as `Authorization: Bearer "
                f"{token}`):",
                flush=True,
            )
        asyncio.run(_run_http_async("127.0.0.1", args.port, token))
    else:
        import asyncio

        asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
