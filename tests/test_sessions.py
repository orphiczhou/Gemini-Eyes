"""End-to-end tests for the REST chat facade against a stubbed client.

Covers the fingerprint-chain session table (server.py) as seen through
api._chat: parent/child key chaining, sanitization effects on the chain,
model-key normalization, plan-blocked models and thinking-level refusal.
All offline — the generate call is replaced by a capture stub, no client
session or network is ever constructed.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

import gemini_mcp.api as api
import gemini_mcp.server as server
from gemini_mcp.gemini_client import DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Harness: fake Request + stubbed generate
# ---------------------------------------------------------------------------


class _FakeRequest:
    def __init__(self, payload: Any):
        self._payload = payload

    async def json(self) -> Any:
        return self._payload


class _StubResult:
    """Minimal stand-in for GenerationResult on the register path."""

    def __init__(self, text: str = "ok", cid: str = "c_1", rid: str = "r_1"):
        self.text = text
        self.thoughts = ""
        self.conversation_id = cid
        self.response_id = rid
        self.candidate_id = "rc_1"
        self.completed = True
        self.metadata: list[Any] = [cid, rid, "", None, None, None, None, None, None, ""]
        self.model = "4 Argon"


@pytest.fixture(autouse=True)
def _clean_session_state():
    """Isolate the module-level session table between tests."""
    server._conv_by_cid.clear()
    server._conv_by_key.clear()
    server._metadata_cache.clear()
    yield
    server._conv_by_cid.clear()
    server._conv_by_key.clear()
    server._metadata_cache.clear()


@pytest.fixture()
def chat_stub(monkeypatch):
    """Replace api._generate_serialized with a capture stub.

    Returns the capture list; each entry is {"prompt", "meta", "model",
    "thinking"} and the stub replies with successive _StubResult texts.
    """
    captured: list[dict[str, Any]] = []
    replies: list[str] = []

    def fake_generate(prompt, metadata=None, files=None, model=None, thinking=None):
        captured.append(
            {
                "prompt": prompt,
                "meta": list(metadata) if metadata is not None else None,
                "model": model,
                "thinking": thinking,
            }
        )
        text = replies[len(captured) - 1] if len(captured) <= len(replies) else "ok"
        return _StubResult(text=text, cid=f"c_{len(captured)}")

    monkeypatch.setattr(api, "_generate_serialized", fake_generate)
    return SimpleNamespace(captured=captured, replies=replies)


def _chat(payload: dict[str, Any]) -> Any:
    """Run api._chat synchronously against a fake request."""
    return asyncio.run(api._chat(_FakeRequest(payload)))


def _body(resp: Any) -> dict[str, Any]:
    return json.loads(resp.body.decode("utf-8"))


# ---------------------------------------------------------------------------
# 1.1 Plan-blocked models are rejected before any upstream traffic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["carbon", "gemini-4-carbon", "4-carbon", "高难任务"])
def test_carbon_names_rejected_up_front(chat_stub, name):
    resp = _chat({"model": name, "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 400
    body = _body(resp)
    assert body["error"]["code"] == "model_unavailable"
    assert "Ultra-only" in body["error"]["message"]
    assert "gemini-argon" in body["error"]["message"]
    assert chat_stub.captured == []  # no upstream call left the building


def test_available_model_still_passes(chat_stub):
    resp = _chat({"model": "gemini-pro", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    assert chat_stub.captured[0]["model"] == "pro"


# ---------------------------------------------------------------------------
# 1.3 Deep Think (level 3) refused with an actionable 400
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [3, "3", "deep", "deep_think", "深层"])
def test_deep_think_refused_with_400(chat_stub, value):
    resp = _chat(
        {"gemini_thinking": value, "messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 400
    body = _body(resp)
    assert body["error"]["code"] == "deep_think_unavailable"
    assert "=1/2/2" in body["error"]["message"]
    assert chat_stub.captured == []  # refused before any upstream work


def test_thinking_high_passes_through(chat_stub):
    resp = _chat(
        {"gemini_thinking": "高", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 200
    assert chat_stub.captured[0]["thinking"] == "高"


# ---------------------------------------------------------------------------
# 1.4 model_key normalization: the gemini- prefix must not break the chain
# ---------------------------------------------------------------------------


def test_model_key_prefix_normalized_across_rounds(chat_stub):
    """Round 1 names the model explicitly, round 2 omits it: same fingerprint."""
    r1 = _chat({"model": "gemini-argon", "messages": [{"role": "user", "content": "u1"}]})
    assert r1.status_code == 200
    r2 = _chat(
        {
            "messages": [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "u2"},
            ]
        }
    )
    assert r2.status_code == 200
    # chain hit: only the last user turn goes upstream, metadata is reused
    assert chat_stub.captured[1]["prompt"] == "u2"
    assert chat_stub.captured[1]["meta"] is not None
    assert chat_stub.captured[1]["meta"][0] == "c_1"


def test_model_key_switch_breaks_chain(chat_stub):
    r1 = _chat({"model": "gemini-argon", "messages": [{"role": "user", "content": "u1"}]})
    assert r1.status_code == 200
    r2 = _chat(
        {
            "model": "gemini-pro",
            "messages": [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "u2"},
            ],
        }
    )
    # different model scope -> miss -> full transcript, new conversation
    assert chat_stub.captured[1]["meta"] is None
    assert "[user]\nu1" in chat_stub.captured[1]["prompt"]
