"""Idle auto-exit (GEMINI_IDLE_TIMEOUT) for the daemon.

All offline: the watchdog check runs directly (no thread, no real sleep),
time is backdated through state.touch_activity(now=...), and os._exit is
stubbed so the test process survives. The tests assert invariants only:
exit ordered / not ordered, counter values, payload fields.
"""

from __future__ import annotations

import json
import logging
import os
import time
from types import SimpleNamespace

import pytest

import gemini_mcp.api as api
import gemini_mcp.server as server
from gemini_mcp import state
from gemini_mcp.gemini_client import GeminiWebClient

from test_sessions import _FakeRequest, _StubResult  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_idle_state(monkeypatch):
    """Isolate the idle clock and the env knob between tests."""
    monkeypatch.delenv("GEMINI_IDLE_TIMEOUT", raising=False)
    state.reset()
    yield
    state.reset()


@pytest.fixture()
def exit_calls(monkeypatch):
    """Stub os._exit (as the watchdog calls it) and record the codes."""
    calls: list[int] = []
    monkeypatch.setattr(os, "_exit", lambda code: calls.append(code))
    return calls


# ---------------------------------------------------------------------------
# (a) idle beyond the timeout -> exit ordered, with a WARNING
# ---------------------------------------------------------------------------


def test_idle_exit_triggered_after_timeout(monkeypatch, exit_calls, caplog):
    monkeypatch.setenv("GEMINI_IDLE_TIMEOUT", "1")  # 1 minute
    state.touch_activity(now=time.time() - 120)  # idle 2 minutes

    with caplog.at_level(logging.WARNING, logger="gemini_mcp"):
        assert server._idle_check_once() is True

    assert exit_calls == [0]  # graceful exit code, ordered exactly once
    warning = next(
        r for r in caplog.records if "Idle auto-exit" in r.getMessage()
    )
    assert "2.0 minutes" in warning.getMessage()  # reports the idle duration
    assert "GEMINI_IDLE_TIMEOUT=1" in warning.getMessage()


def test_idle_clock_counts_from_daemon_start(monkeypatch, exit_calls):
    """No request ever arrived: the clock runs from process start."""
    monkeypatch.setenv("GEMINI_IDLE_TIMEOUT", "0.001")  # 60ms
    time.sleep(0.08)  # cross the (tiny) timeout without touching activity
    assert server._idle_check_once() is True
    assert exit_calls == [0]


# ---------------------------------------------------------------------------
# (b) recent activity prevents the exit
# ---------------------------------------------------------------------------


def test_activity_refresh_prevents_exit(monkeypatch, exit_calls):
    monkeypatch.setenv("GEMINI_IDLE_TIMEOUT", "1")
    state.touch_activity(now=time.time() - 10)  # idle 10s < 60s
    assert server._idle_check_once() is False
    assert exit_calls == []


# ---------------------------------------------------------------------------
# (c) timeout=0 disables the idle exit permanently
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["0", "0.0", "-5"])
def test_timeout_zero_or_negative_disables_exit(monkeypatch, exit_calls, value):
    monkeypatch.setenv("GEMINI_IDLE_TIMEOUT", value)
    state.touch_activity(now=time.time() - 999_999)  # idle ~11.5 days
    assert server._idle_check_once() is False
    assert exit_calls == []


# ---------------------------------------------------------------------------
# (d) /healthz exposes the idle configuration and clock
# ---------------------------------------------------------------------------


def test_healthz_exposes_idle_fields(monkeypatch, exit_calls):
    monkeypatch.setenv("GEMINI_IDLE_TIMEOUT", "45")
    state.touch_activity(now=time.time() - 30)
    monkeypatch.setattr(api, "_get_client", lambda: object())

    payload = api._healthz_payload()
    assert payload["idle_timeout_minutes"] == 45
    assert 25 <= payload["last_activity_seconds_ago"] <= 35
    assert exit_calls == []  # observing the clock does not exit


def test_idle_timeout_env_parsing():
    # unset -> default 30
    assert state.idle_timeout_minutes() == 30
    # junk / empty -> default 30 (never silently disables the exit)
    os.environ["GEMINI_IDLE_TIMEOUT"] = "junk"
    assert state.idle_timeout_minutes() == 30
    os.environ["GEMINI_IDLE_TIMEOUT"] = ""
    assert state.idle_timeout_minutes() == 30
    # numeric (minutes, float allowed)
    os.environ["GEMINI_IDLE_TIMEOUT"] = "90.5"
    assert state.idle_timeout_minutes() == 90.5
    del os.environ["GEMINI_IDLE_TIMEOUT"]


# ---------------------------------------------------------------------------
# what counts as activity (and what must NOT)
# ---------------------------------------------------------------------------


def test_rest_chat_refreshes_activity(monkeypatch):
    """api._chat touches the clock at request start (covers streaming too,
    which never reaches _generate_serialized)."""

    def fake_generate(prompt, metadata=None, files=None, model=None, thinking=None):
        return _StubResult(text="ok")

    monkeypatch.setattr(api, "_generate_serialized", fake_generate)
    state.touch_activity(now=time.time() - 600)

    import asyncio

    resp = asyncio.run(api._chat(_FakeRequest({"messages": [{"role": "user", "content": "hi"}]})))
    assert resp.status_code == 200
    assert state.last_activity_seconds_ago() < 60  # refreshed from ~600s ago


def test_generate_serialized_refreshes_activity(monkeypatch, tmp_path):
    """The MCP tool chokepoint _generate_serialized also touches the clock."""
    cookie_file = tmp_path / "cookies.json"
    cookie_file.write_text(json.dumps({"__Secure-1PSID": "v1"}), encoding="utf-8")
    monkeypatch.setattr(server, "_cookie_file", str(cookie_file))
    monkeypatch.setattr(
        GeminiWebClient,
        "generate",
        lambda self, *a, **kw: SimpleNamespace(conversation_id=None, metadata=None),
    )
    state.touch_activity(now=time.time() - 600)

    server._generate_serialized("hello")
    assert state.last_activity_seconds_ago() < 60


def test_rotation_does_not_refresh_activity(monkeypatch):
    """Spec: cookie rotation is housekeeping, NOT usage — it must never keep
    the daemon alive on its own."""

    class _RotClient:
        session = SimpleNamespace(cookies=[])

        def rotate_cookies(self, force=False, skip_if_fresh=False, cookie_file=None):
            return True

    monkeypatch.setattr(server, "_get_client", lambda: _RotClient())
    monkeypatch.setattr(server, "_cookie_file", None)
    monkeypatch.setattr(server, "_cookies_from_file", False)
    state.touch_activity(now=time.time() - 600)

    server._rotate_once()
    assert 590 <= state.last_activity_seconds_ago() <= 610  # unchanged


def test_healthz_probe_does_not_refresh_activity(monkeypatch):
    """/healthz is the control-plane probe: it observes the clock but must
    not refresh it, or the daemon could never idle out."""
    monkeypatch.setattr(api, "_get_client", lambda: object())
    state.touch_activity(now=time.time() - 600)

    payload = api._healthz_payload()
    assert payload["ok"] is True
    assert 590 <= state.last_activity_seconds_ago() <= 610  # unchanged
    assert 590 <= payload["last_activity_seconds_ago"] <= 610
