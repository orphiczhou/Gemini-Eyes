"""Process-local auth/rotation/idle/credential-health state (leaf module).

server.py owns the background rotation loop, the idle watchdog and the
health watchdog, while api.py — which imports server helpers at module
level (server imports api only lazily) — exposes the counters on /healthz
and /v1/status. Both surfaces import THIS leaf module (same pattern as
usage.py), so no import cycle can arise.

Red line: counters, enums and timestamps only — never cookie values or
credentials (eventlog.py sanitization contract).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Optional

log = logging.getLogger("gemini_mcp")

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

# ---------------------------------------------------------------------------
# Credential health classification (health watchdog, server.py)
#
# healthy          probes pass; nothing to do.
# degraded          2+ consecutive probe failures; auto recovery (dump +
#                  reload + verify) in progress / retrying.
# fetch_blocked    the browser session is HEALTHY (live page holds SNlM0e —
#                  CDP self-check passed) but cookie-replay HTTP token
#                  fetch is blocked (session-device binding, DBSC-class).
#                  NOT a login problem: browser-assisted tokens are the
#                  path back (GEMINI_TOKEN_SOURCE=auto/cdp).
# dead_wait        the dump tool itself failed (exit 1 / timeout); retry is
#                  backed off 30min -> 1h -> 4h (cap). Probing continues.
# dead_needs_login dump succeeds, the session stays dead AND the browser
#                  page is logged out too (CDP self-check failed): a dead
#                  shell only a HUMAN login fixes. Auto-dumping stops
#                  (pointless); probes slow to 30min and reconnect fully
#                  automatically once one passes.
# ---------------------------------------------------------------------------

CREDENTIAL_STATES = (
    "healthy",
    "degraded",
    "fetch_blocked",
    "dead_wait",
    "dead_needs_login",
)
_HEALTH_INTERVAL_DEFAULT_MINUTES = 15.0
_DEAD_PROBE_INTERVAL_MINUTES = 30.0
# dead_wait backoff per consecutive dump failure (minutes), capped at the
# last entry.
_DUMP_BACKOFF_MINUTES = (30.0, 60.0, 240.0)

_credential_state = "healthy"
_last_health_check_at: Optional[float] = None
_probe_fail_streak = 0     # consecutive failed probes (2 => act)
_dump_ok_fail_streak = 0   # consecutive "dump ok but session still dead" cycles
_dump_fail_streak = 0      # consecutive dump failures (backoff ladder)
_next_recovery_at = 0.0    # epoch; recovery attempts gated until then


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
    """Zero all counters and restart the clocks (tests and fresh starts)."""
    global _auth_failures, _last_result, _last_rotation_at
    global _daemon_started_at, _last_activity_at
    global _credential_state, _last_health_check_at
    global _probe_fail_streak, _dump_ok_fail_streak, _dump_fail_streak, _next_recovery_at
    with _lock:
        _auth_failures = 0
        _last_result = None
        _last_rotation_at = None
        _daemon_started_at = time.time()
        _last_activity_at = None
        _credential_state = "healthy"
        _last_health_check_at = None
        _probe_fail_streak = 0
        _dump_ok_fail_streak = 0
        _dump_fail_streak = 0
        _next_recovery_at = 0.0


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


# ---------------------------------------------------------------------------
# Credential health machine API (driven by the health watchdog in server.py)
# ---------------------------------------------------------------------------


def health_check_interval_minutes() -> float:
    """GEMINI_HEALTH_CHECK_INTERVAL_MIN in minutes (default 15; <= 0 disables
    the health watchdog thread). Live env read; junk falls back to default."""
    raw = os.environ.get("GEMINI_HEALTH_CHECK_INTERVAL_MIN")
    if raw is None or not str(raw).strip():
        return _HEALTH_INTERVAL_DEFAULT_MINUTES
    try:
        return float(str(raw).strip())
    except ValueError:
        return _HEALTH_INTERVAL_DEFAULT_MINUTES


def dead_probe_interval_minutes() -> float:
    """Probe cadence while in dead_needs_login (waiting for a manual login)."""
    return _DEAD_PROBE_INTERVAL_MINUTES


def credential_state() -> str:
    with _lock:
        return _credential_state


def probe_fail_streak() -> int:
    with _lock:
        return _probe_fail_streak


def set_credential_state(new: str, detail: str = "") -> bool:
    """Transition the credential state; every CHANGE is event-logged.

    Recoveries to healthy log INFO ("auto-recovered"); entering any broken
    state logs WARNING. Returns True when the state actually changed.
    Never logs cookie values — detail strings carry state/durations only.
    """
    global _credential_state, _dump_ok_fail_streak
    if new not in CREDENTIAL_STATES:
        raise ValueError(f"unknown credential state: {new!r}")
    with _lock:
        old = _credential_state
        if new == old:
            return False
        _credential_state = new
        if new == "healthy":
            _dump_ok_fail_streak = 0
    message = f"credential_state {old} -> {new}"
    if new == "healthy":
        message += " (auto-recovered)" + (f": {detail}" if detail else "")
        log.info(message)
    else:
        message += (f": {detail}" if detail else "")
        log.warning(message)
    return True


def record_probe(ok: bool) -> None:
    """Record one health probe result. Updates last_health_check_at and the
    consecutive-failure streak. Deliberately does NOT touch_activity: the
    daemon must not keep itself alive by checking its own health."""
    global _last_health_check_at, _probe_fail_streak
    with _lock:
        _last_health_check_at = time.time()
        _probe_fail_streak = 0 if ok else _probe_fail_streak + 1


def register_dump_failed() -> float:
    """Count one failed dump (exit 1 / timeout); returns the dead_wait
    backoff in minutes (30 -> 60 -> 240, capped) and gates recovery until
    it elapses."""
    global _dump_fail_streak, _next_recovery_at
    with _lock:
        _dump_fail_streak += 1
        idx = min(_dump_fail_streak, len(_DUMP_BACKOFF_MINUTES)) - 1
        backoff = _DUMP_BACKOFF_MINUTES[idx]
        _next_recovery_at = time.time() + backoff * 60
        return backoff


def register_dump_succeeded() -> None:
    """A dump that ran fine resets the backoff ladder."""
    global _dump_fail_streak
    with _lock:
        _dump_fail_streak = 0


def register_dump_ok_but_still_dead() -> int:
    """Dump succeeded but the session is still dead — the dead-shell signal.
    Returns the consecutive count (2 => dead_needs_login)."""
    global _dump_ok_fail_streak
    with _lock:
        _dump_ok_fail_streak += 1
        return _dump_ok_fail_streak


def defer_recovery(minutes: float) -> None:
    """Gate recovery attempts for the next ``minutes`` minutes."""
    global _next_recovery_at
    with _lock:
        _next_recovery_at = time.time() + minutes * 60


def recovery_deferred(now: Optional[float] = None) -> bool:
    """True while a dead_wait backoff window is still running."""
    with _lock:
        return (now if now is not None else time.time()) < _next_recovery_at


def credential_snapshot() -> dict[str, Any]:
    """Credential-health view for /healthz and /v1/status."""
    with _lock:
        return {
            "credential_state": _credential_state,
            "last_health_check_seconds_ago": (
                round(time.time() - _last_health_check_at, 1)
                if _last_health_check_at is not None
                else None
            ),
            "health_check_interval_minutes": health_check_interval_minutes(),
        }
