"""Tests for the sanitized daemon event log (eventlog.py + REST call sites).

Red line under test: request/session/error events are recorded with model
labels, ids, fingerprint prefixes and character COUNTS only — prompt and
reply bodies must never appear in the file.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

import pytest

import gemini_mcp.api as api
import gemini_mcp.eventlog as eventlog
from gemini_mcp.eventlog import configure_event_logging, event_log_path
from gemini_mcp.gemini_client import GeminiWebError

from test_sessions import _StubResult, _chat, _clean_session_state  # noqa: F401


@pytest.fixture(autouse=True)
def _session_table(_clean_session_state):
    """Activate the session-table isolation from test_sessions here too."""
    yield


@pytest.fixture()
def log_file(tmp_path, monkeypatch):
    path = tmp_path / "sub" / "gemini-mcp.log"  # dir must be auto-created
    monkeypatch.setenv("GEMINI_LOG_FILE", str(path))
    assert configure_event_logging() == path
    yield path
    eventlog.reset_event_logging()


def test_default_path_is_repo_logs_dir(monkeypatch):
    monkeypatch.delenv("GEMINI_LOG_FILE", raising=False)
    path = event_log_path()
    assert path == Path(__file__).resolve().parent.parent / ".logs" / "gemini-mcp.log"


def test_configure_creates_directory(log_file: Path):
    assert log_file.parent.is_dir()
    assert log_file.exists()  # the handler opens (and creates) the file eagerly


def test_request_events_sanitized(log_file, monkeypatch):
    """Two-round chat (miss then hit) through the real _chat flow."""
    captured = []

    def fake_generate(prompt, metadata=None, files=None, model=None, thinking=None):
        captured.append(prompt)
        return _StubResult(text="reply-body-REDACTED")

    monkeypatch.setattr(api, "_generate_serialized", fake_generate)
    _chat({"messages": [{"role": "user", "content": "PROMPT-SECRET-1"}]})
    _chat(
        {
            "messages": [
                {"role": "user", "content": "PROMPT-SECRET-1"},
                {"role": "assistant", "content": "reply-body-REDACTED"},
                {"role": "user", "content": "PROMPT-SECRET-2"},
            ]
        }
    )

    text = log_file.read_text(encoding="utf-8")
    assert "request_start" in text and "stream=False" in text
    assert "session_miss" in text
    assert "session_hit" in text and "conversation_id=c_1" in text
    assert "fingerprint=" in text  # 8-hex prefix logged
    assert "request_end" in text and "chars_in=" in text and "chars_out=" in text
    # RED LINE: no prompt or reply bodies ever hit the file
    assert "PROMPT-SECRET" not in text
    assert "REDACTED" not in text
    # only length figures appear
    assert f"chars_in={len('PROMPT-SECRET-1')}" in text


def test_error_event_logged(log_file, monkeypatch):
    def failing_generate(prompt, metadata=None, files=None, model=None, thinking=None):
        raise GeminiWebError("usage limit", code=1037)

    monkeypatch.setattr(api, "_generate_serialized", failing_generate)
    resp = _chat({"messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 429
    text = log_file.read_text(encoding="utf-8")
    assert "request_error" in text
    assert "code=upstream_rate_limited" in text
    assert "http_status=429" in text
    # the exception message (upstream text) must not leak into the log
    assert "usage limit" not in text


def test_model_fallback_event_logged(log_file):
    from gemini_mcp.gemini_client import GeminiWebClient, resolve_model_setting

    client = GeminiWebClient({})
    client._record_fallback(
        resolve_model_setting("argon"), resolve_model_setting("pro"), 1052
    )
    text = log_file.read_text(encoding="utf-8")
    assert "model_fallback from=argon to=pro code=1052" in text


def test_configure_failure_returns_none(tmp_path, monkeypatch):
    # a directory in place of the log file cannot be opened for writing
    blocked = tmp_path / "blocked.log"
    blocked.mkdir()
    monkeypatch.setenv("GEMINI_LOG_FILE", str(blocked))
    assert configure_event_logging() is None
    eventlog.reset_event_logging()


# ---------------------------------------------------------------------------
# P2-H: the lazy path must give up after one failed configuration
# ---------------------------------------------------------------------------


def test_lazy_configure_fails_once_not_forever(tmp_path, monkeypatch):
    blocked = tmp_path / "blocked.log"
    blocked.mkdir()
    monkeypatch.setenv("GEMINI_LOG_FILE", str(blocked))

    attempts = {"n": 0}
    real_mkdir = eventlog.Path.mkdir

    def counting_mkdir(self, *a, **kw):
        attempts["n"] += 1
        return real_mkdir(self, *a, **kw)

    monkeypatch.setattr(eventlog.Path, "mkdir", counting_mkdir)

    assert configure_event_logging() is None  # attempt 1: fails, sticks
    assert eventlog._lazy_disabled is True
    for _ in range(5):
        eventlog.event_logger()  # lazy: must NOT retry mkdir/open
    assert attempts["n"] == 1

    # explicit reconfigure still retries and recovers (operator fixed env)
    good = tmp_path / "fixed.log"
    monkeypatch.setenv("GEMINI_LOG_FILE", str(good))
    assert configure_event_logging() == good
    assert eventlog._lazy_disabled is False
    eventlog.reset_event_logging()


def test_concurrent_first_touch_mounts_one_handler(tmp_path, monkeypatch):
    """Concurrent lazy configuration must mount exactly one rotating handler
    (a check-then-attach race would double-log and double-hold the file)."""
    import threading

    path = tmp_path / "events.log"
    monkeypatch.setenv("GEMINI_LOG_FILE", str(path))
    eventlog.reset_event_logging()
    barrier = threading.Barrier(8)

    def touch():
        barrier.wait()
        eventlog.event_logger().info("concurrent first touch")

    threads = [threading.Thread(target=touch) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    handlers = [
        h
        for h in logging.getLogger("gemini_mcp").handlers
        if isinstance(h, logging.handlers.RotatingFileHandler)
    ]
    assert len(handlers) == 1
    eventlog.reset_event_logging()


# ---------------------------------------------------------------------------
# P2-F: a mid-stream client disconnect must close the event-log bookkeeping
# ---------------------------------------------------------------------------


def test_stream_disconnect_logs_client_disconnected(tmp_path, monkeypatch):
    """Consumer vanishes mid-SSE (GeneratorExit): no terminal frame is ever
    produced, and the finally block must record code=client_disconnected."""
    import asyncio

    import gemini_mcp.api as api
    from test_sessions import _FakeRequest

    path = tmp_path / "disc.log"
    monkeypatch.setenv("GEMINI_LOG_FILE", str(path))
    assert configure_event_logging() == path

    class _DeltaOnlyClient:
        def generate_stream(self, *args, **kwargs):
            yield {"type": "delta", "text": "partial ", "thoughts": ""}
            # final never arrives before the consumer walks away
            import time as _t

            _t.sleep(30)

    monkeypatch.setattr(api, "_get_client", lambda: _DeltaOnlyClient())

    payload = {"messages": [{"role": "user", "content": "hi"}], "stream": True}
    resp = asyncio.run(api._chat(_FakeRequest(payload)))

    async def abandon_mid_stream():
        gen = resp.body_iterator
        await gen.__anext__()  # role frame
        await gen.__anext__()  # delta frame
        await gen.aclose()  # client disconnect -> GeneratorExit inside sse()

    asyncio.run(abandon_mid_stream())
    text = path.read_text(encoding="utf-8")
    assert "client_disconnected" in text
    assert "request_start" in text
    # and no request_end was fabricated for the abandoned stream
    assert "request_end" not in text
    eventlog.reset_event_logging()


def test_stream_completed_does_not_log_disconnect(tmp_path, monkeypatch):
    """A stream that reaches its final frame must NOT be misreported as a
    client disconnect (the happy path still logs request_end only)."""
    import asyncio

    import gemini_mcp.api as api
    from test_sessions import _FakeRequest, _StubResult

    path = tmp_path / "ok.log"
    monkeypatch.setenv("GEMINI_LOG_FILE", str(path))
    assert configure_event_logging() == path

    class _OkClient:
        def generate_stream(self, *args, **kwargs):
            yield {"type": "final", "result": _StubResult(text="done")}

    monkeypatch.setattr(api, "_get_client", lambda: _OkClient())
    payload = {"messages": [{"role": "user", "content": "hi"}], "stream": True}
    resp = asyncio.run(api._chat(_FakeRequest(payload)))

    async def drain():
        async for _ in resp.body_iterator:
            pass

    asyncio.run(drain())
    text = path.read_text(encoding="utf-8")
    assert "request_end" in text
    assert "client_disconnected" not in text
    eventlog.reset_event_logging()
