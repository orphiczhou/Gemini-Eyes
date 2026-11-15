"""Autonomous health watchdog with dead-profile classification.

All offline: the probe, dump/kill subprocesses and the hot-reload are
stubbed through a harness; time is backdated via state's public knobs
(touch_activity(now=...), defer_recovery(...)); no thread, no real sleep,
no powershell. The tests assert machine invariants: state transitions,
call counts, backoff windows, payload fields.
"""

from __future__ import annotations

import logging
import os
import time

import pytest

import gemini_mcp.api as api
import gemini_mcp.server as server
from gemini_mcp import state

from test_hot_reload import _probe_client, _write_cookies  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_health_state(monkeypatch):
    """Isolate the credential machine and the env knob between tests."""
    monkeypatch.delenv("GEMINI_HEALTH_CHECK_INTERVAL_MIN", raising=False)
    state.reset()
    yield
    state.reset()


class _Harness:
    """Stub probe/dump/kill/reload; script outcomes and record calls."""

    def __init__(self, monkeypatch):
        self.probe_results: list = []  # consumed per probe call; [] -> True
        self.dump_results: list = []   # consumed per dump call; [] -> True
        self.probe_calls = 0
        self.dump_calls = 0
        self.kill_calls: list = []
        self.reload_calls: list = []
        harness = self

        def fake_probe(client):
            harness.probe_calls += 1
            return harness.probe_results.pop(0) if harness.probe_results else True

        def fake_dump():
            harness.dump_calls += 1
            return harness.dump_results.pop(0) if harness.dump_results else True

        monkeypatch.setattr(server, "_probe_credentials", fake_probe)
        monkeypatch.setattr(server, "_run_dump", fake_dump)
        monkeypatch.setattr(
            server, "_run_kill_edge", lambda: harness.kill_calls.append(1) or True
        )
        monkeypatch.setattr(
            server,
            "_maybe_reload_client",
            lambda: harness.reload_calls.append(1) or False,
        )
        monkeypatch.setattr(server, "_get_client", lambda: object())


@pytest.fixture()
def h(monkeypatch):
    return _Harness(monkeypatch)


# ---------------------------------------------------------------------------
# (a) two consecutive failures -> recovery -> ok -> healthy
# ---------------------------------------------------------------------------


def test_two_failures_trigger_recovery_then_healthy(h, caplog):
    h.probe_results = [False, False, True]  # fail, fail, verify-probe ok
    assert server._health_cycle_once() is True  # fail #1: no action yet
    assert h.dump_calls == 0
    assert state.credential_state() == "healthy"

    with caplog.at_level(logging.INFO, logger="gemini_mcp"):
        assert server._health_cycle_once() is True  # fail #2: act

    assert h.dump_calls == 1  # recovery sequence ran (dump -> kill -> verify)
    assert h.kill_calls == [1]
    assert h.reload_calls == [1]
    assert state.credential_state() == "healthy"  # verification probe passed
    messages = [r.getMessage() for r in caplog.records]
    assert any("-> degraded" in m for m in messages)  # WARNING before recovery
    assert any("auto-recovered" in m for m in messages)  # INFO after


def test_single_failure_is_ignored(h):
    """One failed probe (proxy jitter) triggers nothing."""
    h.probe_results = [False]
    server._health_cycle_once()
    assert h.dump_calls == 0
    assert state.credential_state() == "healthy"
    assert state.probe_fail_streak() == 1


# ---------------------------------------------------------------------------
# (b) dump ok but still dead x2 cycles -> dead_needs_login, no more dumps
# ---------------------------------------------------------------------------


def test_dump_ok_but_still_dead_classifies_needs_login(h, caplog):
    h.probe_results = [False] * 10
    h.dump_results = [True] * 10
    with caplog.at_level(logging.INFO, logger="gemini_mcp"):
        server._health_cycle_once()  # fail #1
        server._health_cycle_once()  # fail #2 -> recovery #1, verify fails
    assert state.credential_state() == "degraded"  # 1/2 dead-shell signal

    with caplog.at_level(logging.INFO, logger="gemini_mcp"):
        server._health_cycle_once()  # fail #3 -> recovery #2, verify fails
    assert state.credential_state() == "dead_needs_login"
    assert any("dead_needs_login" in r.getMessage() for r in caplog.records)
    assert h.dump_calls == 2

    server._health_cycle_once()  # dead shell: probing continues, dumping stops
    server._health_cycle_once()
    assert h.dump_calls == 2  # no further auto-dumps
    # probe cadence slowed while waiting for a manual profile login
    assert server._health_probe_interval_minutes() == 30


# ---------------------------------------------------------------------------
# (c) probe passes in a dead state -> immediate re-dump -> healthy
# ---------------------------------------------------------------------------


def test_dead_state_probe_recovery_re_dumps_and_goes_healthy(h, caplog):
    state.set_credential_state("dead_needs_login", detail="seeded for test")
    h.probe_results = [True, True]  # cycle probe + post-dump verify probe

    with caplog.at_level(logging.INFO, logger="gemini_mcp"):
        server._health_cycle_once()

    assert h.dump_calls == 1  # auto re-dump on recovery
    assert h.reload_calls == [1]
    assert state.credential_state() == "healthy"
    assert any("auto-recovered" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# (d) dump failure -> dead_wait backoff 30min -> 1h -> 4h (capped)
# ---------------------------------------------------------------------------


def test_dump_failure_backoff_ladder_unit():
    assert state.register_dump_failed() == 30
    assert state.register_dump_failed() == 60
    assert state.register_dump_failed() == 240
    assert state.register_dump_failed() == 240  # capped
    state.register_dump_succeeded()
    assert state.register_dump_failed() == 30  # ladder resets after a good dump


def test_dump_failure_backoff_through_cycles(h, caplog):
    h.probe_results = [False] * 10
    h.dump_results = [False] * 10
    with caplog.at_level(logging.INFO, logger="gemini_mcp"):
        server._health_cycle_once()  # fail #1: no action
        server._health_cycle_once()  # fail #2 -> dump fails -> dead_wait 30min
    assert state.credential_state() == "dead_wait"
    assert h.dump_calls == 1
    assert state.recovery_deferred(now=time.time() + 29 * 60)
    assert not state.recovery_deferred(now=time.time() + 31 * 60)

    server._health_cycle_once()  # still inside the backoff window
    assert h.dump_calls == 1  # no dump attempt while deferred

    state.defer_recovery(-1)  # expire the window (same knob the machine uses)
    server._health_cycle_once()  # dump fails again -> 1h
    assert state.recovery_deferred(now=time.time() + 59 * 60)
    assert not state.recovery_deferred(now=time.time() + 61 * 60)

    state.defer_recovery(-1)
    server._health_cycle_once()  # third failure -> 4h
    assert state.recovery_deferred(now=time.time() + 239 * 60)
    assert not state.recovery_deferred(now=time.time() + 241 * 60)

    state.defer_recovery(-1)
    server._health_cycle_once()  # fourth failure -> capped at 4h
    assert state.recovery_deferred(now=time.time() + 239 * 60)
    assert not state.recovery_deferred(now=time.time() + 241 * 60)
    assert h.dump_calls == 4
    # dead_wait does not stop probing: a passing probe reconnects anytime
    h.probe_results = [True, True]
    server._health_cycle_once()
    assert state.credential_state() == "healthy"


# ---------------------------------------------------------------------------
# (e) probing/recovery never refresh the idle clock
# ---------------------------------------------------------------------------


def test_probe_and_recovery_do_not_touch_activity(h):
    state.touch_activity(now=time.time() - 600)
    h.probe_results = [False, False, False]  # 2 cycle probes + 1 failed verify
    h.dump_results = [True]

    server._health_cycle_once()
    server._health_cycle_once()  # full recovery sequence runs (dump+kill+verify)

    assert 580 <= state.last_activity_seconds_ago() <= 620  # idle clock unmoved


# ---------------------------------------------------------------------------
# (f) /healthz and /v1/status expose the credential fields
# ---------------------------------------------------------------------------


def test_healthz_exposes_credential_fields(monkeypatch):
    monkeypatch.setenv("GEMINI_HEALTH_CHECK_INTERVAL_MIN", "7")
    state.set_credential_state("degraded", detail="seeded for test")
    state.record_probe(False)
    monkeypatch.setattr(api, "_get_client", lambda: object())

    payload = api._healthz_payload()
    assert payload["credential_state"] == "degraded"
    assert payload["health_check_interval_minutes"] == 7
    assert payload["last_health_check_seconds_ago"] is not None
    assert payload["last_health_check_seconds_ago"] < 5


def test_status_payload_exposes_credential_fields(h, monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_HEALTH_CHECK_INTERVAL_MIN", "7")
    cookie_file = tmp_path / "cookies.json"
    _write_cookies(cookie_file, "v1")
    monkeypatch.setattr(server, "_cookie_file", str(cookie_file))
    monkeypatch.setattr(server, "_get_client", _probe_client())
    state.set_credential_state("dead_wait", detail="seeded for test")

    payload = server._status_payload()
    assert payload["credential_state"] == "dead_wait"
    assert payload["health_check_interval_minutes"] == 7
    assert payload["last_health_check_seconds_ago"] is None  # no probe ran yet


# ---------------------------------------------------------------------------
# (g) generate lock held -> the round (probe AND recovery) is skipped
# ---------------------------------------------------------------------------


def test_cycle_skipped_while_generate_lock_held(h):
    # GENERATE_LOCK is an RLock: same-thread acquire(blocking=False) would
    # succeed re-entrantly, so hold it from ANOTHER thread — exactly the
    # production situation (a generate in flight while the watchdog wakes).
    import threading

    held = threading.Event()
    release = threading.Event()

    def hold_lock():
        with server._GENERATE_LOCK:
            held.set()
            release.wait(timeout=10)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    assert held.wait(timeout=5)
    try:
        assert server._health_cycle_once() is False  # round skipped
    finally:
        release.set()
        holder.join(timeout=10)

    assert h.probe_calls == 0  # neither the probe...
    assert h.dump_calls == 0   # ...nor any recovery action started
    assert state.credential_state() == "healthy"
    assert state.credential_snapshot()["last_health_check_seconds_ago"] is None


# ---------------------------------------------------------------------------
# env parsing for the interval knob
# ---------------------------------------------------------------------------


def test_health_interval_env_parsing():
    assert state.health_check_interval_minutes() == 15  # default
    os.environ["GEMINI_HEALTH_CHECK_INTERVAL_MIN"] = "0"
    assert state.health_check_interval_minutes() == 0
    os.environ["GEMINI_HEALTH_CHECK_INTERVAL_MIN"] = "junk"
    assert state.health_check_interval_minutes() == 15
    os.environ["GEMINI_HEALTH_CHECK_INTERVAL_MIN"] = ""
    assert state.health_check_interval_minutes() == 15
    del os.environ["GEMINI_HEALTH_CHECK_INTERVAL_MIN"]
