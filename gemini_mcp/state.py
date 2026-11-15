"""Process-local auth/rotation/idle state (leaf module).

server.py owns the background rotation loop and the idle watchdog, while
api.py — which imports server helpers at module level (server imports api
only lazily) — exposes the counters on /healthz and /v1/status. Both
surfaces import THIS leaf module (same pattern as usage.py), so no import
cycle can arise.

Red line: counters, enums and timestamps only — never cookie values or
credentials (eventlog.py sanitization contract).
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional

_lock = threading.Lock()
_auth_failures = 0
# "ok" / "failed" after the first background rotation; None until then.
_last_result: Optional[str] = None
_last_rotation_at: Optional[float] = None

# Idle auto-exit (GEMINI_IDLE_TIMEOUT): default 30 minutes; <= 0 disables.
_IDLE_TIMEOUT_DEFAULT_MINUTES = 30.0
# Last real request (chat generate, tool call, status/usage/models probe).
# None until the first request; the watchdog then measures from process
# start, so a daemon nobody ever talks to still exits after the timeout.
_daemon_started_at = time.time()
_last_activity_at: Optional[float] = None


def record_rotation(ok: bool) -> int:
    """Record one background-rotation outcome; returns the failure count.

    Failures accumulate (consecutive rejects => growing auth_failures); the
    next success clears the counter — a recovered session must not drag its
    outage history along (the event log keeps the long tail).
    """
    global _auth_failures, _last_result, _last_rotation_at
    with _lock:
        _auth_failures = 0 if ok else _auth_failures + 1
        _last_result = "ok" if ok else "failed"
        _last_rotation_at = time.time()
        return _auth_failures


def reset() -> None:
    """Zero all counters and restart the idle clock (tests and fresh starts)."""
    global _auth_failures, _last_result, _last_rotation_at
    global _daemon_started_at, _last_activity_at
    with _lock:
        _auth_failures = 0
        _last_result = None
        _last_rotation_at = None
        _daemon_started_at = time.time()
        _last_activity_at = None


def snapshot() -> dict[str, Any]:
    """Consistent copy for the /healthz and /v1/status surfaces."""
    with _lock:
        return {
            "auth_failures": _auth_failures,
            "last_result": _last_result,
            "last_rotation_seconds_ago": (
                round(time.time() - _last_rotation_at) if _last_rotation_at else None
            ),
        }


def touch_activity(now: Optional[float] = None) -> None:
    """Mark "a real request just started" (chat generate, tool call,
    status/usage/models probe — REST or MCP). Called at request START so a
    long generation never trips the idle exit mid-flight. Cookie rotation
    and /healthz probes deliberately do NOT touch this: they are internal
    housekeeping, not usage — otherwise the daemon could never idle out."""
    global _last_activity_at
    with _lock:
        _last_activity_at = now if now is not None else time.time()


def last_activity_seconds_ago(now: Optional[float] = None) -> float:
    """Seconds since the last real request; before the first request, seconds
    since the daemon start (module reset)."""
    with _lock:
        base = _last_activity_at if _last_activity_at is not None else _daemon_started_at
        return (now if now is not None else time.time()) - base


def idle_timeout_minutes() -> float:
    """GEMINI_IDLE_TIMEOUT in minutes (default 30; <= 0 disables the exit).

    Read live from the environment each call. Unparseable values fall back
    to the default rather than disabling the exit silently."""
    raw = os.environ.get("GEMINI_IDLE_TIMEOUT")
    if raw is None or not str(raw).strip():
        return _IDLE_TIMEOUT_DEFAULT_MINUTES
    try:
        return float(str(raw).strip())
    except ValueError:
        return _IDLE_TIMEOUT_DEFAULT_MINUTES
