"""Cookie-file hot reload + rotation-failure observability (Wave C).

All offline: the client singleton is rebuilt from a tmp cookie file, the
generate/rotate paths are replaced by stubs, and the observability counters
are reset between tests. No network, no daemon, no real Google session.
"""

from __future__ import annotations

import json
import logging
import os
import time
from types import SimpleNamespace

import pytest

import gemini_mcp.server as server
from gemini_mcp import state
from gemini_mcp.gemini_client import GeminiWebClient


@pytest.fixture(autouse=True)
def _clean_client_state(monkeypatch):
    """Isolate the process-level client singleton, reload baseline and counters."""
    monkeypatch.setattr(server, "_client", None)
    monkeypatch.setattr(server, "_cookies_baseline_at", 0.0)
    monkeypatch.setattr(server, "_cookies_from_file", False)
    monkeypatch.delenv("GEMINI_COOKIE_FILE", raising=False)
    state.reset()
    yield
    state.reset()


def _write_cookies(path, sid: str, newer_by_seconds: float = 0.0) -> None:
    """Write a minimal cookie file; optionally force mtime into the future."""
    path.write_text(json.dumps({"__Secure-1PSID": sid}), encoding="utf-8")
    if newer_by_seconds:
        stamp = time.time() + newer_by_seconds
        os.utime(path, (stamp, stamp))


# ---------------------------------------------------------------------------
# (a) external replacement -> next generate runs on a rebuilt client
# ---------------------------------------------------------------------------


def test_hot_reload_rebuilds_on_newer_mtime(monkeypatch, tmp_path):
    cookie_file = tmp_path / "cookies.json"
    _write_cookies(cookie_file, "v1")
    monkeypatch.setattr(server, "_cookie_file", str(cookie_file))

    first = server._get_client()
    assert first.get_cookie("__Secure-1PSID") == "v1"

    _write_cookies(cookie_file, "v2", newer_by_seconds=5)  # external dump tool
    assert server._maybe_reload_client() is True
    second = server._get_client()
    assert second is not first
    assert second.get_cookie("__Secure-1PSID") == "v2"


def test_generate_serialized_picks_up_new_client(monkeypatch, tmp_path):
    """The _generate_serialized entry check swaps the singleton mid-flight."""
    cookie_file = tmp_path / "cookies.json"
    _write_cookies(cookie_file, "v1")
    monkeypatch.setattr(server, "_cookie_file", str(cookie_file))

    seen = []

    def fake_generate(self, *args, **kwargs):
        seen.append(self)
        return SimpleNamespace(conversation_id=None, metadata=None)

    monkeypatch.setattr(GeminiWebClient, "generate", fake_generate)

    server._generate_serialized("hello")
    _write_cookies(cookie_file, "v2", newer_by_seconds=5)
    server._generate_serialized("hello again")

    assert len(seen) == 2
    assert seen[1] is not seen[0]
    assert seen[1].get_cookie("__Secure-1PSID") == "v2"


# ---------------------------------------------------------------------------
# (b) unchanged mtime -> no rebuild
# ---------------------------------------------------------------------------


def test_no_rebuild_when_mtime_unchanged(monkeypatch, tmp_path):
    cookie_file = tmp_path / "cookies.json"
    _write_cookies(cookie_file, "v1")
    monkeypatch.setattr(server, "_cookie_file", str(cookie_file))

    first = server._get_client()
    assert server._maybe_reload_client() is False
    assert server._get_client() is first


def test_no_reload_without_cookie_file(monkeypatch, tmp_path):
    """Browser-extraction session (no resolvable file path): nothing to watch."""
    monkeypatch.setattr(server, "_cookie_file", None)
    monkeypatch.setattr(server, "DEFAULT_COOKIE_FILE", tmp_path / "absent.json")
    monkeypatch.setattr(
        server, "_client", GeminiWebClient({"__Secure-1PSID": "x"})
    )
    assert server._maybe_reload_client() is False


def test_persist_advances_baseline_no_self_reload(monkeypatch, tmp_path):
    """Rotation's own write-back bumps the file mtime; the baseline must follow
    so the next hot path does not treat our write as an external replacement
    (read-your-own-write loop)."""
    cookie_file = tmp_path / "cookies.json"
    _write_cookies(cookie_file, "v1")
    monkeypatch.setattr(server, "_cookie_file", str(cookie_file))
    monkeypatch.setattr(server, "_cookies_from_file", True)

    client = server._get_client()
    server._persist_cookies(client)  # rewrites the file (mtime moves to now)
    assert server._maybe_reload_client() is False
    assert server._get_client() is client


# ---------------------------------------------------------------------------
# (c) broken replacement keeps the old client
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("content", ["not json", "{}", "[]"])
def test_broken_replacement_keeps_old_client(monkeypatch, tmp_path, content, caplog):
    cookie_file = tmp_path / "cookies.json"
    _write_cookies(cookie_file, "v1")
    monkeypatch.setattr(server, "_cookie_file", str(cookie_file))
    first = server._get_client()

    cookie_file.write_text(content, encoding="utf-8")
    stamp = time.time() + 5
    os.utime(cookie_file, (stamp, stamp))

    with caplog.at_level(logging.WARNING, logger="gemini_mcp"):
        assert server._maybe_reload_client() is False
    assert server._get_client() is first
    assert first.get_cookie("__Secure-1PSID") == "v1"
    assert any("hot-reload failed" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# (d) rotation failures: counter + WARNING, cleared by a success
# ---------------------------------------------------------------------------


class _RotateStubClient:
    """Client stub whose rotate_cookies outcome is scripted per test."""

    session = SimpleNamespace(cookies=[])  # _persist_cookies sees no entries

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)

    def rotate_cookies(self, force=False, skip_if_fresh=False, cookie_file=None):
        return self._outcomes.pop(0)


def _rotate_fixture(monkeypatch, outcomes):
    stub = _RotateStubClient(outcomes)
    monkeypatch.setattr(server, "_get_client", lambda: stub)
    monkeypatch.setattr(server, "_cookie_file", None)
    monkeypatch.setattr(server, "_cookies_from_file", False)
    return stub


def test_rotate_failure_counts_and_warns(monkeypatch, caplog):
    _rotate_fixture(monkeypatch, [False, False])
    with caplog.at_level(logging.WARNING, logger="gemini_mcp"):
        server._rotate_once()
        server._rotate_once()

    snap = state.snapshot()
    assert snap["auth_failures"] == 2
    assert snap["last_result"] == "failed"
    assert snap["last_rotation_seconds_ago"] is not None
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "rotation FAILED" in r.getMessage()
    ]
    assert len(warnings) == 2
    assert "auth_failures=1" in warnings[0].getMessage()
    assert "auth_failures=2" in warnings[1].getMessage()


def test_rotate_success_clears_failure_counter(monkeypatch):
    _rotate_fixture(monkeypatch, [False, True])
    server._rotate_once()
    assert state.snapshot()["auth_failures"] == 1
    server._rotate_once()
    snap = state.snapshot()
    assert snap["auth_failures"] == 0
    assert snap["last_result"] == "ok"


# ---------------------------------------------------------------------------
# (e) /v1/status and /healthz expose the new fields
# ---------------------------------------------------------------------------


def _probe_client():
    class _ProbeClient:
        language = "zh-CN"
        model_description = "4 Argon"
        session_id = "sid"
        _last_rotate_at = 0.0

        def init(self, force=False):
            pass

    return _ProbeClient()


def test_status_payload_exposes_rotation_fields(monkeypatch, tmp_path):
    cookie_file = tmp_path / "cookies.json"
    _write_cookies(cookie_file, "v1")
    monkeypatch.setattr(server, "_cookie_file", str(cookie_file))
    monkeypatch.setattr(server, "_get_client", _probe_client)

    payload = server._status_payload()
    assert payload["ok"] is True
    assert isinstance(payload["auth_failures"], int) and payload["auth_failures"] == 0
    assert payload["cookie_rotation"]["last_result"] is None  # no rotation yet
    assert payload["cookie_rotation"]["interval_seconds"] == server._ROTATE_INTERVAL


def test_status_payload_fields_survive_broken_cookies(monkeypatch, tmp_path):
    """The failure branches keep the fields — the recovery orchestrator probes
    exactly when things are broken."""
    cookie_file = tmp_path / "cookies.json"
    cookie_file.write_text("not json", encoding="utf-8")
    monkeypatch.setattr(server, "_cookie_file", str(cookie_file))

    state.record_rotation(False)
    payload = server._status_payload()
    assert payload["ok"] is False
    assert payload["auth_failures"] == 1
    assert payload["cookie_rotation"]["last_result"] == "failed"


def test_healthz_exposes_auth_failures(monkeypatch):
    from gemini_mcp import api

    state.record_rotation(False)
    state.record_rotation(False)
    monkeypatch.setattr(api, "_get_client", lambda: object())

    payload = api._healthz_payload()
    assert payload["ready"] is True
    assert payload["auth_failures"] == 2


def test_healthz_auth_failures_present_when_not_ready(monkeypatch):
    from gemini_mcp import api

    def _boom():
        raise RuntimeError("no cookies")

    monkeypatch.setattr(api, "_get_client", _boom)

    payload = api._healthz_payload()
    assert payload["ok"] is False
    assert payload["ready"] is False
    assert payload["auth_failures"] == 0  # field exists even before any rotation
