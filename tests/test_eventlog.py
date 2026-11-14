"""Tests for the sanitized daemon event log (eventlog.py + REST call sites).

Red line under test: request/session/error events are recorded with model
labels, ids, fingerprint prefixes and character COUNTS only — prompt and
reply bodies must never appear in the file.
"""

from __future__ import annotations

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
