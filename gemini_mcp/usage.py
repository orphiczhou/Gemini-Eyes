"""Process-local usage counters shared by the MCP tools and the REST layer.

The web backend exposes no per-account token/usage API, so "how much have I
used" is answered locally: every generate() call (成功或失败) bumps these
counters, and model-chain degradations (Argon -> Pro -> Flash when the
account hits 1037 usage limits) are recorded by the client as the practical
"plan headroom" signal. Snapshot shape is stable JSON for both surfaces
(``gemini_usage`` MCP tool and ``GET /v1/usage``).
"""

import threading
import time
from typing import Any

_lock = threading.Lock()
_started_at = time.time()

_counters: dict[str, int] = {
    "requests_total": 0,
    "requests_ok": 0,
    "requests_error": 0,
    "chars_in": 0,
    "chars_out": 0,
}


def record_request(chars_in: int = 0, chars_out: int = 0) -> None:
    """Count one completed (successful) generate call."""
    with _lock:
        _counters["requests_total"] += 1
        _counters["requests_ok"] += 1
        _counters["chars_in"] += max(0, int(chars_in))
        _counters["chars_out"] += max(0, int(chars_out))


def record_error() -> None:
    """Count one failed generate call (transport, auth, upstream errors)."""
    with _lock:
        _counters["requests_total"] += 1
        _counters["requests_error"] += 1


def snapshot() -> dict[str, Any]:
    """Consistent copy of the counters plus process uptime."""
    with _lock:
        data = dict(_counters)
    data["started_at"] = _started_at
    data["uptime_seconds"] = round(time.time() - _started_at, 1)
    # Rough character-based size estimate; the web backend has no token
    # accounting, so this is an order-of-magnitude figure only (~4 chars/token).
    data["tokens_in_est"] = _counters["chars_in"] // 4
    data["tokens_out_est"] = _counters["chars_out"] // 4
    return data
