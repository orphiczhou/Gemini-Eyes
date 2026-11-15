"""Browser-assisted token source (CDP) and the fetch_blocked classification.

All offline: the DevTools HTTP/WS layers are stubbed at module seams
(_http_json, _evaluate, _CDPSocket, subprocess.Popen), the token-source
policy is driven through GEMINI_TOKEN_SOURCE, and the health-watchdog
classification reuses the harness from test_health_watchdog. No real Edge,
no real CDP, no network.
"""

from __future__ import annotations

import logging

import pytest

import gemini_mcp.browser_token as browser_token
import gemini_mcp.server as server
from gemini_mcp import state
from gemini_mcp.gemini_client import GeminiWebClient, GeminiWebError

from test_health_watchdog import _Harness  # noqa: E402

SENTINEL = "XYZ-TOKEN-SENTINEL-42"

# Captured at import: the autouse guard below replaces _ensure_cdp_ready,
# so the spawn-gating test needs the pristine function back.
_REAL_ENSURE_CDP_READY = browser_token._ensure_cdp_ready


@pytest.fixture(autouse=True)
def _no_real_browser(monkeypatch):
    """Default stub: the browser layer is inert unless a test overrides it —
    no test may ever spawn a real Edge or touch port 9333. Stub the readiness
    seam (not the public function) so extraction tests can still run the
    real get_browser_token against stubbed /json + evaluate."""
    monkeypatch.setattr(browser_token, "_ensure_cdp_ready", lambda spawn=True: False)
    monkeypatch.delenv("GEMINI_TOKEN_SOURCE", raising=False)
    yield


@pytest.fixture(autouse=True)
def _clean_credential_state():
    """Isolate the credential machine between tests (probe streaks and
    dump counters would otherwise leak across the scripted sequences)."""
    state.reset()
    yield
    state.reset()


@pytest.fixture()
def h(monkeypatch):
    """The health-watchdog harness (same stubs as test_health_watchdog)."""
    return _Harness(monkeypatch)


# ---------------------------------------------------------------------------
# WebSocket client frames (pure byte-level, no socket)
# ---------------------------------------------------------------------------


class _RecordingSock:
    def __init__(self):
        self.sent = b""

    def sendall(self, data):
        self.sent += data


def _socket_on_fake_wire():
    sock = object.__new__(browser_token._CDPSocket)
    sock._timeout = 5.0
    sock._sock = _RecordingSock()
    sock._buffer = b""
    return sock


def test_ws_client_text_frame_is_masked():
    sock = _socket_on_fake_wire()
    sock._send_frame(b"abc")
    wire = sock._sock.sent
    assert wire[0] == 0x81  # FIN + text opcode
    assert wire[1] == 0x80 | 3  # MASK bit + length 3
    mask = wire[2:6]
    payload = bytes(b ^ mask[i % 4] for i, b in enumerate(b"abc"))
    assert wire[6:] == payload


def test_ws_recv_assembles_text_and_answers_ping():
    import os as _os

    sock = _socket_on_fake_wire()
    # one ping frame (server->client, unmasked) then one text frame
    ping_payload = b"hb"
    ping = bytes([0x89, len(ping_payload)]) + ping_payload
    text_payload = b'{"id":1}'
    text = bytes([0x81, len(text_payload)]) + text_payload
    sock._buffer = ping + text
    message = sock._recv_message()
    assert message == '{"id":1}'
    # the ping was answered with a pong carrying the same payload
    pong = sock._sock.sent
    assert pong[0] == 0x8A and pong[1] == 0x80 | len(ping_payload)


# ---------------------------------------------------------------------------
# CDP extraction (stubbed /json + Runtime.evaluate)
# ---------------------------------------------------------------------------


def _stub_cdp(monkeypatch, tabs, evaluate_value, opened=None):
    calls = {"paths": [], "new": [], "evaluate": []}

    def fake_http_json(path, method="GET", timeout=3.0):
        calls["paths"].append((path, method))
        if path == "/json/version":
            return {"Browser": "Edge"}
        if path == "/json":
            return tabs
        if path.startswith("/json/new"):
            calls["new"].append(method)
            return opened
        raise AssertionError(f"unexpected CDP path {path}")

    def fake_evaluate(ws_url, expression, timeout=10.0):
        calls["evaluate"].append((ws_url, expression))
        return evaluate_value

    monkeypatch.setattr(browser_token, "_http_json", fake_http_json)
    monkeypatch.setattr(browser_token, "_evaluate", fake_evaluate)
    monkeypatch.setattr(browser_token, "_ensure_cdp_ready", lambda spawn=True: True)
    return calls


def test_get_browser_token_prefers_existing_gemini_tab(monkeypatch):
    tabs = [
        {"type": "page", "url": "https://example.org/", "webSocketDebuggerUrl": "ws://x/1"},
        {
            "type": "page",
            "url": "https://gemini.google.com/app",
            "webSocketDebuggerUrl": "ws://127.0.0.1:9333/devtools/page/ABC",
        },
    ]
    calls = _stub_cdp(monkeypatch, tabs, SENTINEL)
    assert browser_token.get_browser_token() == SENTINEL
    assert not calls["new"]  # existing tab reused, no /json/new
    ws_url, expression = calls["evaluate"][0]
    assert ws_url.endswith("/devtools/page/ABC")
    assert "SNlM0e" in expression  # the extraction regex runs in the page


def test_get_browser_token_opens_tab_when_none(monkeypatch):
    calls = _stub_cdp(
        monkeypatch,
        tabs=[{"type": "page", "url": "https://example.org/", "webSocketDebuggerUrl": "ws://x/1"}],
        evaluate_value=SENTINEL,
        opened={"type": "page", "url": "https://gemini.google.com/app", "webSocketDebuggerUrl": "ws://127.0.0.1:9333/devtools/page/NEW"},
    )
    assert browser_token.get_browser_token() == SENTINEL
    assert calls["new"] == ["PUT"]  # modern CDP requires PUT for /json/new
    assert calls["evaluate"][0][0].endswith("/devtools/page/NEW")


def test_get_browser_token_logged_out_page(monkeypatch):
    _stub_cdp(
        monkeypatch,
        tabs=[{"type": "page", "url": "https://gemini.google.com/app", "webSocketDebuggerUrl": "ws://127.0.0.1:9333/devtools/page/ABC"}],
        evaluate_value=None,  # live page holds no SNlM0e -> logged out
    )
    assert browser_token.get_browser_token() is None


def test_token_value_never_logged(monkeypatch, caplog):
    _stub_cdp(
        monkeypatch,
        tabs=[{"type": "page", "url": "https://gemini.google.com/app", "webSocketDebuggerUrl": "ws://x/y"}],
        evaluate_value=SENTINEL,
    )
    with caplog.at_level(logging.INFO, logger="gemini_mcp"):
        token = browser_token.get_browser_token()
    assert token == SENTINEL
    for record in caplog.records:
        assert SENTINEL not in record.getMessage()  # value stays in memory
    # but the acquisition itself IS observable (length only)
    assert any("len=" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Edge spawn (argv construction only — never a real spawn)
# ---------------------------------------------------------------------------


def test_edge_spawn_args_flags(monkeypatch):
    monkeypatch.setattr(browser_token, "_edge_executable", lambda: r"C:\Edge\msedge.exe")
    argv = browser_token._edge_spawn_args()
    assert argv[0] == r"C:\Edge\msedge.exe"
    assert "--remote-debugging-port=9333" in argv
    assert any(
        arg.startswith("--user-data-dir=") and arg.endswith("edge-mcp-profile")
        for arg in argv
    )
    assert "--no-first-run" in argv
    assert "--no-default-browser-check" in argv
    assert "https://gemini.google.com/app" in argv
    assert not any("--headless" in arg for arg in argv)  # visible window per spec


def test_ensure_cdp_ready_spawn_gating(monkeypatch):
    monkeypatch.setattr(browser_token, "_ensure_cdp_ready", _REAL_ENSURE_CDP_READY)
    spawned = []

    class _FakeProc:
        pass

    monkeypatch.setattr(
        browser_token.subprocess,
        "Popen",
        lambda argv, **kw: spawned.append(argv) or _FakeProc(),
    )
    monkeypatch.setattr(browser_token, "_edge_executable", lambda: "msedge")
    monkeypatch.setattr(browser_token, "_cdp_ready", lambda: False)
    monkeypatch.setattr(browser_token, "_wait_cdp_ready", lambda timeout=15.0: True)

    assert browser_token._ensure_cdp_ready(spawn=True) is True
    assert len(spawned) == 1  # spawn attempted exactly once, args captured
    assert spawned[0][0] == "msedge"
    assert browser_token._ensure_cdp_ready(spawn=False) is False  # no spawn
    assert len(spawned) == 1


# ---------------------------------------------------------------------------
# GEMINI_TOKEN_SOURCE policy branches (through _fetch_tokens)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, text):
        self.status_code = 200
        self.text = text


_SHELL_TEXT = (
    'window.WIZ_global_data = {"cfb2h":"bl_x","FdrFJe":"sid_x",'
    '"TuX5cc":"zh","qKIAYe":"pid"}'  # shell: cfb2h present, SNlM0e STRIPPED
)
_TOKEN_TEXT = (
    '{"SNlM0e":"HTTP_TOKEN","cfb2h":"bl_x","FdrFJe":"sid_x",'
    '"TuX5cc":"zh","qKIAYe":"pid"}'
)


class _FakeSession:
    def __init__(self, response):
        self._response = response
        self.cookies = type(
            "J", (), {"set": lambda *a, **k: None, "__iter__": lambda s: iter(())}
        )()
        self.headers = {}

    def get(self, url, **kw):
        return self._response


def _client_with_page(text):
    client = GeminiWebClient({})
    client.session = _FakeSession(_FakeResponse(text))
    return client


@pytest.fixture()
def browser_calls(monkeypatch):
    """Recorder stub for the browser layer (overrides the inert default)."""
    calls = {"n": 0, "token": SENTINEL}

    def fake_get(spawn=True):
        calls["n"] += 1
        return calls["token"]

    monkeypatch.setattr(browser_token, "get_browser_token", fake_get)
    return calls


def test_auto_policy_uses_http_token_when_present(monkeypatch, browser_calls):
    monkeypatch.setenv("GEMINI_TOKEN_SOURCE", "auto")
    client = _client_with_page(_TOKEN_TEXT)
    client._fetch_tokens()
    assert client.access_token == "HTTP_TOKEN"
    assert client.token_source == "http"
    assert browser_calls["n"] == 0  # no browser involved


def test_auto_policy_falls_back_to_browser_on_shell(monkeypatch, browser_calls):
    monkeypatch.setenv("GEMINI_TOKEN_SOURCE", "auto")
    client = _client_with_page(_SHELL_TEXT)  # 200 shell, SNlM0e stripped
    client._fetch_tokens()
    assert browser_calls["n"] == 1  # exactly one CDP attempt
    assert client.access_token == SENTINEL
    assert client.token_source == "cdp"
    assert client.build_label == "bl_x"  # aux values still parsed from the shell
    assert client.session_id == "sid_x"


def test_auto_policy_old_error_when_browser_also_fails(monkeypatch, browser_calls):
    monkeypatch.setenv("GEMINI_TOKEN_SOURCE", "auto")
    browser_calls["token"] = None  # CDP fails too
    client = _client_with_page(_SHELL_TEXT)
    with pytest.raises(GeminiWebError) as excinfo:
        client._fetch_tokens()
    assert "Could not parse SNlM0e" in str(excinfo.value)
    assert "(CDP) also failed" in str(excinfo.value)


def test_http_policy_never_touches_browser(monkeypatch, browser_calls):
    monkeypatch.setenv("GEMINI_TOKEN_SOURCE", "http")
    client = _client_with_page(_SHELL_TEXT)
    with pytest.raises(GeminiWebError) as excinfo:
        client._fetch_tokens()
    assert "Could not parse SNlM0e" in str(excinfo.value)
    assert "CDP" not in str(excinfo.value)  # exactly the old error, no hint
    assert browser_calls["n"] == 0


def test_cdp_policy_prefers_browser_even_when_http_token_exists(monkeypatch, browser_calls):
    monkeypatch.setenv("GEMINI_TOKEN_SOURCE", "cdp")
    client = _client_with_page(_TOKEN_TEXT)  # http token present but pinned to browser
    client._fetch_tokens()
    assert client.access_token == SENTINEL
    assert client.token_source == "cdp"
    assert browser_calls["n"] == 1


def test_cdp_policy_raises_when_browser_fails(monkeypatch, browser_calls):
    monkeypatch.setenv("GEMINI_TOKEN_SOURCE", "cdp")
    browser_calls["token"] = None
    client = _client_with_page(_TOKEN_TEXT)
    with pytest.raises(GeminiWebError) as excinfo:
        client._fetch_tokens()
    assert "CDP) unavailable" in str(excinfo.value)


def test_token_source_policy_parsing(monkeypatch):
    monkeypatch.delenv("GEMINI_TOKEN_SOURCE", raising=False)
    assert browser_token.token_source_policy() == "auto"  # default
    for value in ("http", "cdp", "auto"):
        monkeypatch.setenv("GEMINI_TOKEN_SOURCE", value)
        assert browser_token.token_source_policy() == value
    monkeypatch.setenv("GEMINI_TOKEN_SOURCE", "junk")
    assert browser_token.token_source_policy() == "auto"  # junk -> auto


# ---------------------------------------------------------------------------
# fetch_blocked classification: CDP self-check BEFORE any 'needs login'
# ---------------------------------------------------------------------------


def _run_dead_cycles(harness, cycles=3, verify_after_classify=None):
    """Drive the machine to the 'dump ok but still dead x2' classification."""
    # probe consumption: c1 = 1 (streak builds, no recovery); every later
    # cycle = 2 (cycle probe + recovery verify probe)
    harness.probe_results = [False] * (1 + 2 * (cycles - 1))
    if verify_after_classify is not None:
        harness.probe_results.append(verify_after_classify)
    harness.dump_results = [True] * cycles
    for _ in range(cycles):
        server._health_cycle_once()


def test_fetch_blocked_when_browser_session_healthy(h, caplog):
    h.login_results = [SENTINEL]  # live page signed in
    with caplog.at_level(logging.INFO, logger="gemini_mcp"):
        _run_dead_cycles(h, cycles=3, verify_after_classify=True)

    assert state.credential_state() == "healthy"  # browser-assisted path healed
    assert h.login_calls  # the CDP self-check ran and decided
    transitions = [r.getMessage() for r in caplog.records]
    assert any("fetch_blocked" in m for m in transitions)
    assert any("auto-recovered" in m for m in transitions)
    assert not any("dead_needs_login" in m for m in transitions)  # no misreport
    assert not any(SENTINEL in m for m in transitions)  # token value never logged


def test_fetch_blocked_persists_when_browser_token_not_usable(h):
    h.login_results = [SENTINEL]
    _run_dead_cycles(h, cycles=3, verify_after_classify=False)
    # classified fetch_blocked (NOT dead_needs_login) and stayed there
    assert state.credential_state() == "fetch_blocked"


def test_fetch_blocked_not_downgraded_on_later_cycles(h):
    h.login_results = [SENTINEL, SENTINEL]  # classify runs again on c4
    _run_dead_cycles(h, cycles=3, verify_after_classify=False)
    assert state.credential_state() == "fetch_blocked"
    # another failing cycle re-runs recovery + classification but keeps
    # the fetch_blocked verdict (browser still healthy)
    h.probe_results = [False, False, False]  # probe, verify, classify-verify
    h.dump_results = [True]
    server._health_cycle_once()
    assert state.credential_state() == "fetch_blocked"


def test_self_check_precedes_dead_needs_login_verdict(h, monkeypatch):
    order = []
    original_set = state.set_credential_state

    def recording_set(new, detail=""):
        order.append(("state", new))
        return original_set(new, detail)

    monkeypatch.setattr(state, "set_credential_state", recording_set)
    h.login_results = [None]  # browser page logged out too
    _run_dead_cycles(h, cycles=3)
    # the login self-check ran exactly once, and the dead_needs_login
    # verdict is the LAST transition — it could only follow the check
    assert h.login_calls == [1]
    assert order[-1] == ("state", "dead_needs_login")
    assert state.credential_state() == "dead_needs_login"


def test_healthz_passes_fetch_blocked_through(monkeypatch):
    from gemini_mcp import api

    state.set_credential_state("fetch_blocked", detail="seeded for test")
    monkeypatch.setattr(api, "_get_client", lambda: object())
    payload = api._healthz_payload()
    assert payload["credential_state"] == "fetch_blocked"
    assert state.CREDENTIAL_STATES == (
        "healthy",
        "degraded",
        "fetch_blocked",
        "dead_wait",
        "dead_needs_login",
    )
