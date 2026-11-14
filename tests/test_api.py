"""Synthetic tests for the REST facade and per-request model overrides.

All offline: pure payload helpers, the error translation table and the
per-request fallback-chain rebuild. No network, no real client session.
"""

import json

import pytest

from gemini_mcp.api import (
    _ERROR_MAP,
    _gemini_error_response,
    _messages_to_prompt,
    _resolve_thinking,
)
from gemini_mcp.gemini_client import DeepThinkUnavailable, GeminiWebClient, GeminiWebError


# ---------------------------------------------------------------------------
# _messages_to_prompt
# ---------------------------------------------------------------------------


def test_messages_single_turn():
    prompt = _messages_to_prompt([{"role": "user", "content": "hi"}], False)
    assert prompt == "hi"


def test_messages_system_prefix():
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    prompt = _messages_to_prompt(msgs, False)
    assert prompt.startswith("<system instructions>be brief</system instructions>")
    assert prompt.endswith("hi")


def test_messages_multiturn_transcript():
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    prompt = _messages_to_prompt(msgs, False)
    assert "[user]\na" in prompt
    assert "[assistant]\nb" in prompt
    assert prompt.endswith("[assistant]")


def test_messages_continue_sends_last_user_only():
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    assert _messages_to_prompt(msgs, True) == "c"


def test_messages_parts_array_content():
    msgs = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}],
        }
    ]
    assert _messages_to_prompt(msgs, False) == "part1\npart2"


# ---------------------------------------------------------------------------
# thinking resolution
# ---------------------------------------------------------------------------


def test_thinking_explicit_wins_over_effort():
    assert _resolve_thinking({"gemini_thinking": 1, "reasoning_effort": "high"}) == 1
    assert _resolve_thinking({"gemini_thinking": "高"}) == "高"


@pytest.mark.parametrize("value", [3, "3", "deep", "deep_think", "深层"])
def test_thinking_level_3_refused_at_validation(value):
    with pytest.raises(DeepThinkUnavailable):
        _resolve_thinking({"gemini_thinking": value})


def test_thinking_effort_mapping():
    assert _resolve_thinking({"reasoning_effort": "minimal"}) == 1
    assert _resolve_thinking({"reasoning_effort": "low"}) == 1
    assert _resolve_thinking({"reasoning_effort": "medium"}) == 2
    assert _resolve_thinking({"reasoning_effort": "high"}) == 2
    assert _resolve_thinking({}) is None


# ---------------------------------------------------------------------------
# error translation (no upstream text ever echoed)
# ---------------------------------------------------------------------------


def test_error_response_maps_all_codes():
    for code, (status, _name, _msg) in _ERROR_MAP.items():
        resp = _gemini_error_response(GeminiWebError("should not leak", code=code))
        assert resp.status_code == status


def test_error_response_transport_and_generic():
    assert _gemini_error_response(GeminiWebError("x", transport=True)).status_code == 502
    assert _gemini_error_response(GeminiWebError("x")).status_code == 503


def test_error_response_stall_maps_to_504():
    resp = _gemini_error_response(GeminiWebError("upstream silent", stall=True))
    assert resp.status_code == 504
    body = json.loads(resp.body.decode("utf-8"))
    assert body["error"]["code"] == "upstream_stalled"
    assert "Ultra-only" in body["error"]["message"]
    # transport=True + stall=True still reports as a stall (watchdog verdict wins)
    both = _gemini_error_response(GeminiWebError("x", transport=True, stall=True))
    assert both.status_code == 504


# ---------------------------------------------------------------------------
# /v1/models payload: plan blocklist + protocol capabilities
# ---------------------------------------------------------------------------


def test_models_payload_carbon_blocked_and_capabilities(monkeypatch):
    from types import SimpleNamespace

    from gemini_mcp import api

    monkeypatch.setattr(
        api, "_get_client", lambda: SimpleNamespace(model_registry_snapshot=lambda: None)
    )
    payload = api._models_payload()
    carbon = next(m for m in payload["data"] if m["id"] == "gemini-carbon")
    assert carbon["available"] is False
    assert "Ultra-only" in carbon["unavailable_reason"]
    # servable entries carry no availability marker at all
    others = [m for m in payload["data"] if m["id"] != "gemini-carbon"]
    assert others and all("available" not in m for m in others)
    assert payload["capabilities"] == {"function_calling": False}
    # aliases advertise the honest target: UI names for carbon point at carbon
    assert payload["gemini"]["model_aliases"]["4-carbon"] == "carbon"
    assert payload["gemini"]["model_aliases"]["高难任务"] == "carbon"
    # thinking guide: 低/中/高 = 1/2/2, Deep Think marked unavailable
    guide = {row["label"]: row for row in payload["gemini"]["thinking_guide"]}
    assert guide["低"]["value"] == 1
    assert guide["中"]["value"] == 2
    assert guide["高"]["value"] == 2
    assert guide["Deep Think"]["value"] == 3
    assert guide["Deep Think"]["api"] == "不可用"
    assert payload["gemini"]["thinking_levels"]["高"] == 2
    assert "deep" not in payload["gemini"]["thinking_levels"]


def test_carbon_alias_resolution():
    from gemini_mcp.gemini_client import _normalise_model_name

    assert _normalise_model_name("carbon") == "carbon"
    assert _normalise_model_name("gemini-4-carbon") == "carbon"
    assert _normalise_model_name("4-carbon") == "carbon"
    assert _normalise_model_name("高难任务") == "carbon"


# ---------------------------------------------------------------------------
# per-request model chain rebuild
# ---------------------------------------------------------------------------


def _client(model=None, thinking=None) -> GeminiWebClient:
    return GeminiWebClient({"__Secure-1PSID": "x"}, model=model, thinking=thinking)


def test_chain_override_model_and_thinking():
    client = _client("argon", 2)
    chain = client._build_model_chain("flash", 1)
    assert chain[0].label == "flash"
    assert chain[0].thinking == 1
    # the instance default chain is untouched
    assert client.model_chain[0].label == "argon"
    assert client.model_chain[0].thinking == 2


def test_chain_override_thinking_only_keeps_model():
    client = _client("pro", 2)
    chain = client._build_model_chain(None, 1)
    assert chain[0].label == "pro"
    assert chain[0].thinking == 1


def test_chain_noop_returns_default():
    client = _client()
    assert client._build_model_chain(None, None) is client.model_chain


def test_chain_legacy_request():
    client = _client("argon", 2)
    assert client._build_model_chain("legacy", None) == [None]


def test_chain_always_ends_with_legacy_fallback():
    client = _client()
    chain = client._build_model_chain("flash-lite", None)
    assert chain[-1] is None
    assert all(setting is None or hasattr(setting, "label") for setting in chain)
