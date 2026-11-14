"""Process-local auth/rotation observability state (leaf module).

server.py owns the background rotation loop, while api.py — which imports
server helpers at module level (server imports api only lazily) — exposes
the counters on /healthz and /v1/status. Both surfaces import THIS leaf
module (same pattern as usage.py), so no import cycle can arise.

Red line: counters, enums and timestamps only — never cookie values or
credentials (eventlog.py sanitization contract).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

_lock = threading.Lock()
_auth_failures = 0
# "ok" / "failed" after the first background rotation; None until then.
_last_result: Optional[str] = None
_last_rotation_at: Optional[float] = None


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
    """Zero all counters (tests and fresh starts)."""
    global _auth_failures, _last_result, _last_rotation_at
    with _lock:
        _auth_failures = 0
        _last_result = None
        _last_rotation_at = None


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
