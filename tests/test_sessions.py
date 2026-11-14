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
import time
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
# 1.8 Message sanitization: tool traffic never reaches prompts or fingerprints
# ---------------------------------------------------------------------------


def test_tool_history_stays_out_of_transcript_and_chain(chat_stub):
    """A tool-speaking agent's replay must chain like a plain chat: tool
    roles/payloads vanish from the prompt, the fingerprint ignores them."""
    r1 = _chat({"messages": [{"role": "user", "content": "list files"}]})
    assert r1.status_code == 200

    r2 = _chat(
        {
            "messages": [
                {"role": "user", "content": "list files"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "run_ls", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "content": "TOOL-OUTPUT-PAYLOAD", "tool_call_id": "call_1"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "now delete one"},
            ]
        }
    )
    assert r2.status_code == 200
    prompt = chat_stub.captured[1]["prompt"]
    # chain hit: sanitized history equals the plain-chat history
    assert prompt == "now delete one"
    assert chat_stub.captured[1]["meta"] is not None
    assert chat_stub.captured[1]["meta"][0] == "c_1"
    # nothing tool-shaped leaked upstream
    assert "TOOL-OUTPUT-PAYLOAD" not in prompt
    assert "run_ls" not in prompt
    assert "tool" not in prompt.lower()


def test_all_tool_messages_rejected_as_empty(chat_stub):
    resp = _chat({"messages": [{"role": "tool", "content": "only tool output"}]})
    assert resp.status_code == 400
    assert chat_stub.captured == []


def test_request_level_tool_fields_accepted_but_ignored(chat_stub):
    resp = _chat(
        {
            "model": "gemini-pro",
            "tools": [{"type": "function", "function": {"name": "run_ls"}}],
            "tool_choice": "auto",
            "functions": [{"name": "legacy_fn"}],
            "function_call": "none",
            "parallel_tool_calls": False,
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert resp.status_code == 200
    assert chat_stub.captured[0]["prompt"] == "hi"


# ---------------------------------------------------------------------------
# SSE streaming through api._chat_stream
# ---------------------------------------------------------------------------


def _chat_stream_collect(payload: dict[str, Any]) -> str:
    """Run _chat with a streaming body and collect the full SSE text."""
    resp = asyncio.run(api._chat(_FakeRequest(payload)))

    async def collect() -> str:
        chunks = []
        async for c in resp.body_iterator:
            chunks.append(c if isinstance(c, str) else c.decode("utf-8"))
        return "".join(chunks)

    return asyncio.run(collect())


def test_stream_stall_surfaces_as_upstream_stalled(monkeypatch):
    """A stalled upstream ends the SSE stream with the mapped 504 error."""
    from gemini_mcp.gemini_client import GeminiWebError

    class _StallClient:
        def generate_stream(self, *args, **kwargs):
            raise GeminiWebError("upstream silent", stall=True)
            yield  # noqa: B901 - unreachable; makes this a generator function

    monkeypatch.setattr(api, "_get_client", lambda: _StallClient())
    body = _chat_stream_collect(
        {"messages": [{"role": "user", "content": "hi"}], "stream": True}
    )
    assert "upstream_stalled" in body
    assert "Ultra-only" in body
    assert body.rstrip().endswith("data: [DONE]")


def test_stream_final_registers_session_and_chains(monkeypatch, chat_stub):
    """A finished streamed turn registers the child key; the follow-up
    request chains onto it (streaming counterpart of the model_key fix)."""

    class _OkClient:
        def generate_stream(self, prompt, metadata=None, model=None, thinking=None, cancel=None, **kw):
            yield {"type": "delta", "text": "streamed ", "thoughts": ""}
            yield {"type": "final", "result": _StubResult(text="streamed reply")}

    monkeypatch.setattr(api, "_get_client", lambda: _OkClient())
    body = _chat_stream_collect(
        {"messages": [{"role": "user", "content": "u1"}], "stream": True}
    )
    assert "streamed " in body  # delta reached the wire
    assert "[DONE]" in body
    assert '"finish_reason": "stop"' in body

    # follow-up non-streaming turn chains onto the streamed registration
    resp = _chat(
        {
            "messages": [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "streamed reply"},
                {"role": "user", "content": "u2"},
            ]
        }
    )
    assert resp.status_code == 200
    assert chat_stub.captured[0]["prompt"] == "u2"
    assert chat_stub.captured[0]["meta"] is not None
    assert chat_stub.captured[0]["meta"][0] == "c_1"


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


# ---------------------------------------------------------------------------
# 1.6 Fingerprint chain: hit sends the last user turn only, miss replays all
# ---------------------------------------------------------------------------


def test_chain_hit_sends_last_user_turn_and_reuses_metadata(chat_stub):
    r1 = _chat({"messages": [{"role": "user", "content": "u1"}]})
    assert r1.status_code == 200
    # fresh conversation: single-turn prompt, no metadata
    assert chat_stub.captured[0]["prompt"] == "u1"
    assert chat_stub.captured[0]["meta"] is None

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
    # hit: only the last user turn is sent, the web-side conversation continues
    assert chat_stub.captured[1]["prompt"] == "u2"
    assert chat_stub.captured[1]["meta"] is not None
    assert chat_stub.captured[1]["meta"][0] == "c_1"
    assert chat_stub.captured[1]["meta"][1] == "r_1"


def test_chain_miss_sends_full_transcript_as_new_conversation(chat_stub):
    _chat({"messages": [{"role": "user", "content": "u1"}]})
    # the replayed assistant turn was rewritten -> parent fingerprint differs
    r2 = _chat(
        {
            "messages": [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "REWRITTEN"},
                {"role": "user", "content": "u2"},
            ]
        }
    )
    assert r2.status_code == 200
    assert chat_stub.captured[1]["meta"] is None
    prompt = chat_stub.captured[1]["prompt"]
    assert "[user]\nu1" in prompt
    assert "[assistant]\nREWRITTEN" in prompt
    assert prompt.endswith("[assistant]")


def test_system_add_remove_keeps_chain_intact(chat_stub):
    """system messages never enter the fingerprint — editing them mid-chat
    must neither break nor create a chain."""
    _chat({"messages": [{"role": "system", "content": "be brief"},
                        {"role": "user", "content": "u1"}]})
    r2 = _chat({"messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "u2"},
    ]})
    assert r2.status_code == 200
    # still chained: only the new user turn (plus the system prefix, which is
    # always re-sent) — no transcript replay
    assert chat_stub.captured[1]["prompt"].endswith("u2")
    assert "[user]" not in chat_stub.captured[1]["prompt"]

    # system message dropped entirely in round 3 -> still chained
    r3 = _chat({"messages": [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "u3"},
    ]})
    assert r3.status_code == 200
    assert chat_stub.captured[2]["prompt"] == "u3"


# ---------------------------------------------------------------------------
# 1.6 Session table internals: parent/child keys, eviction, orphan cleanup
# ---------------------------------------------------------------------------


class _RegResult:
    def __init__(self, cid, rid="r_x"):
        self.conversation_id = cid
        self.metadata = [cid, rid, "", None, None, None, None, None, None, ""]


def _register(cid="c_a", texts=("user: u1", "assistant: ok"), model="argon"):
    server.conv_register(_RegResult(cid), list(texts), model)


def test_conv_register_parent_and_child_keys():
    _register()
    parent = server.session_key(["user: u1"], "argon")
    child = server.session_key(["user: u1", "assistant: ok"], "argon")
    assert server._conv_by_key[parent] == "c_a"
    assert server._conv_by_key[child] == "c_a"
    # lookup paths: parent fingerprint (next request) and explicit id
    assert server.conv_lookup(["user: u1"], "argon") is not None
    assert server.conv_lookup([], "argon", conv_id="c_a") is not None


def test_session_key_stable_and_model_scoped():
    a = server.session_key(["user: u1", "assistant: ok"], "argon")
    b = server.session_key(["user: u1", "assistant: ok"], "argon")
    assert a == b  # deterministic
    assert a != server.session_key(["user: u1", "assistant: ok"], "pro")  # model-scoped
    assert a != server.session_key(["user: u1", "assistant: ok!"], "argon")  # text-scoped
    assert a != server.session_key(["assistant: ok", "user: u1"], "argon")  # order-scoped


def test_conv_register_idempotent_for_retries():
    """A client retrying the identical request re-registers the same keys."""
    _register()
    _register()
    assert len(server._conv_by_key) == 2  # parent + child, not duplicated
    assert server.conv_lookup(["user: u1"], "argon") is not None


def test_conv_evict_ttl():
    _register()
    server._conv_by_cid["c_a"].updated_at = time.time() - server._CONV_TTL - 1
    assert server.conv_lookup(["user: u1"], "argon") is None  # expired
    server._conv_evict()
    assert "c_a" not in server._conv_by_cid
    assert not server._conv_by_key  # keys died with the record


def test_conv_evict_capacity_trims_oldest(monkeypatch):
    monkeypatch.setattr(server, "_CONV_MAX", 2)
    now = time.time()
    for i, cid in enumerate(("c_old", "c_mid", "c_new")):
        _register(cid=cid, texts=(f"user: q{i}",))
        server._conv_by_cid[cid].updated_at = now - (100 - i)  # distinct ages
    server._conv_evict()
    assert set(server._conv_by_cid) == {"c_mid", "c_new"}  # oldest trimmed
    # surviving keys still resolve, trimmed ones are gone
    assert server.conv_lookup(["user: q2"], "argon") is not None
    assert server.conv_lookup(["user: q0"], "argon") is None


def test_conv_evict_cleans_orphan_keys_unconditionally():
    """Regression: orphan keys used to survive when nothing was stale/over
    capacity — they poisoned later lookups with dead conversation ids."""
    _register()  # healthy record, table under capacity, nothing stale
    server._conv_by_key["deadbeef"] = "c_gone"  # orphan: no cid record
    server._conv_evict()
    assert "deadbeef" not in server._conv_by_key
    # the healthy entries are untouched
    assert server.conv_lookup(["user: u1"], "argon") is not None


def test_conv_eviction_forced_on_every_register(monkeypatch):
    """Deterministic half of P2-G: _CONV_MAX pinned small so every register
    walks the eviction path — the newest records survive, everything the
    trimmed records contributed is cleaned up."""
    monkeypatch.setattr(server, "_CONV_MAX", 4)
    for i in range(10):
        _register(cid=f"c_{i}", texts=(f"user: q{i}",))
    assert len(server._conv_by_cid) == 4
    assert set(server._conv_by_cid) == {"c_6", "c_7", "c_8", "c_9"}
    # newest by-id lookups hit; evicted ones miss
    assert server.conv_lookup([], "argon", conv_id="c_9") is not None
    assert server.conv_lookup([], "argon", conv_id="c_0") is None
    # orphan-free: every surviving fingerprint key points at a live record
    assert server._conv_by_key
    for key_cid in server._conv_by_key.values():
        assert key_cid in server._conv_by_cid


def test_session_table_thread_smoke(monkeypatch):
    """P2-G: hammer register/lookup with _CONV_MAX=4 so EVERY register runs
    the eviction path concurrently. A lock defect surfaces as an exception,
    a cross-wired lookup record, or an orphan fingerprint key (a key pointing
    at a conversation id that no longer has a record) observed under the lock.
    """
    import threading

    monkeypatch.setattr(server, "_CONV_MAX", 4)
    errors: list[str] = []

    def worker(n: int) -> None:
        try:
            for i in range(60):
                cid = f"c_{n}_{i}"
                _register(cid=cid, texts=(f"user: q{n}_{i}",))
                rec = server.conv_lookup([f"user: q{n}_{i}"], "argon", conv_id=cid)
                if rec is not None and str((rec.metadata or [""])[0] or "") != cid:
                    errors.append(f"cross-wired record for {cid}")
                # orphan-free invariant, sampled under the same lock the table
                # uses (a torn snapshot here means the lock is not held on
                # some compound path)
                with server._CONV_LOCK:
                    orphans = [
                        k
                        for k, v in server._conv_by_key.items()
                        if v not in server._conv_by_cid
                    ]
                    if orphans:
                        errors.append(f"orphan keys under lock: {orphans[:3]}")
        except Exception as exc:  # noqa: BLE001 - record and fail
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(server._conv_by_cid) <= server._CONV_MAX  # capacity respected
    # orphan-free at quiesce too
    for key_cid in server._conv_by_key.values():
        assert key_cid in server._conv_by_cid


# ---------------------------------------------------------------------------
# γ-1: MCP initialize serverInfo.version is populated
# ---------------------------------------------------------------------------


def test_mcp_serverinfo_version_populated():
    from gemini_mcp import __version__
    from gemini_mcp.server import mcp

    assert __version__  # resolved from dist metadata (or synced constant)
    assert mcp.version == __version__
    assert mcp.name == "gemini-web"
