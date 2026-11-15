"""Browser-assisted SNlM0e acquisition over CDP (read-only, stdlib only).

Field evidence (differential experiment, 2026-11): with a fully healthy
dedicated Edge profile (page renders, in-page fetch works, live page holds
SNlM0e), replaying the SAME cookies through python HTTP fetches the app
shell (HTTP 200, 922KB, WIZ_global_data present) with SNlM0e stripped;
curl_cffi chrome145 impersonation fails identically — so it is not a TLS
fingerprint issue. Conclusion: session-device binding (DBSC-class) —
cookie-replay clients are no longer issued session tokens. The only token
source is a REAL browser page. This module extracts the token from the live
page via the DevTools protocol.

RED LINES
* Token values live in memory only — never logged, never written to disk.
* CDP is strictly read-only here: it evaluates one expression on the page.
  It never writes cookies.json (the daemon stays the single writer).
* The spawned Edge is KEPT ALIVE after extraction (visible window, no
  --headless — headless risk-control behaviour is unverified): later token
  refreshes reuse the same browser. Its in-page cookie rotation stays
  inside the profile and does not touch cookies.json.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import socket
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("gemini_mcp")

_CDP_HOST = "127.0.0.1"
_CDP_PORT = 9333
_CDP_READY_TIMEOUT = 15.0  # spec: readiness probe <= 15s after spawn
_GEMINI_APP_URL = "https://gemini.google.com/app"

# Dedicated profile (same location tools/mcp-browser-profile.ps1 uses).
_PROFILE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "edge-mcp-profile"

_EDGE_CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)

# Runs IN the page: regex the token out of the live DOM. cfb2h et al. stay
# on the http path (the shell still carries them).
_SNL_EXTRACT_EXPR = (
    '(()=>{const m=document.documentElement.innerHTML'
    '.match(/"SNlM0e":"([^"]+)"/);return m?m[1]:null;})()'
)


def token_source_policy() -> str:
    """GEMINI_TOKEN_SOURCE: auto (default) | http | cdp. Junk falls back to
    auto. auto = http parse first; on the DBSC "shell without token" shape
    (page 200 but SNlM0e stripped) fall back to the browser."""
    raw = (os.environ.get("GEMINI_TOKEN_SOURCE") or "").strip().lower()
    return raw if raw in ("auto", "http", "cdp") else "auto"


# ---------------------------------------------------------------------------
# CDP HTTP endpoints (urllib)
# ---------------------------------------------------------------------------


def _http_json(path: str, method: str = "GET", timeout: float = 3.0) -> Any:
    """One JSON request against the DevTools HTTP endpoint. Raises on any
    transport/status/parse problem — callers treat CDP as best-effort."""
    url = f"http://{_CDP_HOST}:{_CDP_PORT}{path}"
    request = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _cdp_ready() -> bool:
    try:
        _http_json("/json/version")
        return True
    except (OSError, ValueError):
        return False


def _wait_cdp_ready(timeout: float = _CDP_READY_TIMEOUT) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _cdp_ready():
            return True
        time.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
# Edge lifecycle (spawn the dedicated profile; never kill it here)
# ---------------------------------------------------------------------------


def _edge_executable() -> Optional[str]:
    for candidate in _EDGE_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return shutil.which("msedge")


def _edge_spawn_args() -> list[str]:
    """Spawn argv, flags mirroring tools/mcp-browser-profile.ps1.

    No --headless on purpose: unverified against risk control; the spec
    wants a visible window. The browser outlives the token call (and the
    daemon): nothing here ever terminates it."""
    exe = _edge_executable() or "msedge"
    return [
        exe,
        f"--remote-debugging-port={_CDP_PORT}",
        f"--user-data-dir={_PROFILE_DIR}",
        "--no-first-run",
        "--no-default-browser-check",
        _GEMINI_APP_URL,
    ]


def _spawn_edge() -> bool:
    argv = _edge_spawn_args()
    if _edge_executable() is None:
        log.warning("browser token: msedge executable not found")
        return False
    try:
        # Plain Popen: the browser must keep running after we move on
        # (token refresh reuses it); we never wait() or terminate() it.
        subprocess.Popen(argv, close_fds=True)
    except OSError as e:
        log.warning("browser token: cannot spawn msedge: %s", e)
        return False
    log.info(
        "browser token: spawned dedicated-profile Edge (visible window, kept alive)"
    )
    return True


def _ensure_cdp_ready(spawn: bool = True) -> bool:
    if _cdp_ready():
        return True
    if not spawn or not _spawn_edge():
        return False
    return _wait_cdp_ready()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


def _find_gemini_tab() -> Optional[dict[str, Any]]:
    try:
        targets = _http_json("/json")
    except (OSError, ValueError):
        return None
    for target in targets if isinstance(targets, list) else []:
        if (
            target.get("type") == "page"
            and "gemini.google.com" in str(target.get("url", ""))
            and target.get("webSocketDebuggerUrl")
        ):
            return target
    return None


def _open_gemini_tab() -> Optional[dict[str, Any]]:
    """Open a gemini tab via /json/new (PUT on modern Edges; GET fallback)."""
    url = urllib.parse.quote(_GEMINI_APP_URL, safe="")
    for method in ("PUT", "GET"):
        try:
            target = _http_json(f"/json/new?url={url}", method=method)
        except (OSError, ValueError):
            continue
        if isinstance(target, dict) and target.get("webSocketDebuggerUrl"):
            return target
    return None


# ---------------------------------------------------------------------------
# Minimal WebSocket client (RFC 6455, client side only, text frames)
# ---------------------------------------------------------------------------


class _CDPSocket:
    """Just enough WebSocket for one CDP command: HTTP upgrade, masked
    client text frames, unmasked server frames, ping->pong, close raises."""

    def __init__(self, host: str, port: int, path: str, timeout: float = 10.0):
        self._timeout = timeout
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._sock.settimeout(timeout)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        handshake = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self._sock.sendall(handshake.encode("ascii"))
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("CDP handshake ended early (EOF)")
            buf += chunk
        status_line = buf.split(b"\r\n", 1)[0]
        if b"101" not in status_line:
            raise ConnectionError(f"CDP handshake refused: {status_line[:60]!r}")
        self._buffer = buf.split(b"\r\n\r\n", 1)[1]

    # -- raw helpers ------------------------------------------------------

    def _read_exact(self, n: int) -> bytes:
        while len(self._buffer) < n:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("CDP socket EOF")
            self._buffer += chunk
        data, self._buffer = self._buffer[:n], self._buffer[n:]
        return data

    def _send_frame(self, payload: bytes, opcode: int = 0x1) -> None:
        header = bytearray([0x80 | opcode])  # FIN + opcode
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header += length.to_bytes(2, "big")
        else:
            header.append(0x80 | 127)
            header += length.to_bytes(8, "big")
        mask = os.urandom(4)  # client frames MUST be masked
        header += mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self._sock.sendall(bytes(header) + masked)

    def _recv_message(self) -> str:
        """Assemble one text message; transparently answers pings."""
        parts: list[bytes] = []
        while True:
            first_two = self._read_exact(2)
            b1, b2 = first_two[0], first_two[1]
            fin = bool(b1 & 0x80)
            opcode = b1 & 0x0F
            length = b2 & 0x7F
            if length == 126:
                length = int.from_bytes(self._read_exact(2), "big")
            elif length == 127:
                length = int.from_bytes(self._read_exact(8), "big")
            if b2 & 0x80:  # server frames should be unmasked; tolerate anyway
                mask = self._read_exact(4)
                payload = bytes(
                    b ^ mask[i % 4] for i, b in enumerate(self._read_exact(length))
                )
            else:
                payload = self._read_exact(length)
            if opcode == 0x8:
                raise ConnectionError("CDP socket closed by peer")
            if opcode == 0x9:  # ping -> pong
                self._send_frame(payload, opcode=0xA)
                continue
            if opcode in (0x1, 0x0):  # text / continuation
                parts.append(payload)
                if fin:
                    return b"".join(parts).decode("utf-8", "replace")
            # other opcodes (binary, pong) are ignored

    # -- CDP command ------------------------------------------------------

    def command(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one command; return its response (events without the id are
        skipped). The socket timeout bounds the whole exchange."""
        self._send_frame(json.dumps(payload).encode("utf-8"))
        deadline = time.time() + self._timeout
        while time.time() < deadline:
            message = json.loads(self._recv_message())
            if message.get("id") == payload.get("id"):
                return message
        raise TimeoutError("CDP command response timeout")

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


def _evaluate(ws_url: str, expression: str, timeout: float = 10.0) -> Optional[str]:
    """Runtime.evaluate on a page target; returns the string value or None."""
    parts = urllib.parse.urlsplit(ws_url)
    sock = _CDPSocket(
        parts.hostname or _CDP_HOST,
        parts.port or _CDP_PORT,
        parts.path or "/",
        timeout=timeout,
    )
    try:
        response = sock.command(
            {
                "id": 1,
                "method": "Runtime.evaluate",
                "params": {"expression": expression, "returnByValue": True},
            }
        )
    finally:
        sock.close()
    if response.get("error"):
        log.info("browser token: CDP error: %s", response["error"])
        return None
    result = (response.get("result") or {}).get("result") or {}
    value = result.get("value")
    return value if isinstance(value, str) and value else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_browser_token(spawn: bool = True) -> Optional[str]:
    """Extract SNlM0e from the LIVE gemini page over CDP.

    Spawns the dedicated-profile Edge when none is listening (visible
    window, <= 15s readiness wait) and KEEPS it running afterwards. Returns
    the token (memory only — never logged) or None on any failure. A page
    that renders without SNlM0e (logged out / still loading) also yields
    None, which is exactly the login-state signal the health watchdog
    needs: token present == truly signed in."""
    try:
        if not _ensure_cdp_ready(spawn=spawn):
            log.info(
                "browser token: CDP unavailable (Edge not running; spawn=%s)", spawn
            )
            return None
        target = _find_gemini_tab() or _open_gemini_tab()
        if not target or not target.get("webSocketDebuggerUrl"):
            log.info("browser token: no gemini tab available")
            return None
        token = _evaluate(target["webSocketDebuggerUrl"], _SNL_EXTRACT_EXPR)
        if token:
            # Length only — the VALUE never reaches a log line (red line).
            log.info("browser token: SNlM0e acquired via CDP (len=%d)", len(token))
        else:
            log.info("browser token: live page holds no SNlM0e (logged out or loading)")
        return token or None
    except Exception as e:  # noqa: BLE001 - token source is best-effort
        log.warning("browser token: CDP extraction failed: %s", e)
        return None
