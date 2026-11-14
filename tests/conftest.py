"""Shared test configuration: keep the daemon event log out of the repo.

api._chat touches event_logger() on every request, which would lazily create
<repo>/.logs/gemini-mcp.log during tests. Redirect it to the per-test tmp
dir instead, and detach the rotating handler afterwards (Windows keeps
deleted temp files locked while a handler holds them open).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_event_log(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_LOG_FILE", str(tmp_path / "events.log"))
    from gemini_mcp.eventlog import reset_event_logging

    reset_event_logging()  # force re-configuration with the new env path
    yield
    reset_event_logging()
