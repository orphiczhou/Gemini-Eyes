"""Synthetic tests for the account RPCs (model discovery / quota / usage windows).

All offline: bodies are synthetic batchexecute frames fed through the real
parsers via a mocked batch_execute.
"""

import json

from gemini_mcp.gemini_client import GeminiWebClient


def _client() -> GeminiWebClient:
    return GeminiWebClient({"__Secure-1PSID": "x"})


def _frames(rpc_id: str, body) -> list:
    return [["wrb.fr", rpc_id, json.dumps(body), None, "generic"]]


def _mock_batch(client, rpc_id: str, body) -> None:
    client.batch_execute = lambda rpcs, source_path="/app": _frames(rpc_id, body)


def test_fetch_quota_parses_buckets():
    c = _client()
    body = [
        [
            [[[1, 11], 11], None, 0.42, [1794639000], 200, 116],
            [[[1, 4], 4], None, 0.9, [1794700000], 100, 10],
        ]
    ]
    _mock_batch(c, "qpEbW", body)
    quotas = c.fetch_quota()
    by_label = {q["label"]: q for q in quotas.values()}
    assert by_label["Gemini Flash"]["usage_percentage"] == 0.42
    assert by_label["Gemini Flash"]["remaining"] == 116
    assert by_label["Gemini Pro"]["remaining"] == 10
    assert by_label["Gemini Pro"]["total"] == 100


def test_fetch_usage_info_parses_windows():
    c = _client()
    body = [
        2,
        [
            [85, 0.5, 1, [[1794639000]]],
            [990, 0.1, 2, [[1795200000]]],
            [12, None, 3, None],
        ],
        True,
    ]
    _mock_batch(c, "jSf9Qc", body)
    info = c.fetch_usage_info()
    assert info["tier"]["id"] == 2
    assert info["tier"]["label"] == "PRO"
    assert info["current_5h"]["usage_percentage"] == 50
    assert info["weekly"]["remaining_credits"] == 990
    assert info["ai_credits_remaining"] == 12
    assert info["use_overage_ai_credits"] is True


def test_discover_models_parses_roster_and_capacity():
    c = _client()
    body = [None] * 15
    body.append(
        [
            [
                "e6fa609c3fa255c0", "Pro", None, None, None, None, None, None,
                None, None, "Pro", "Gemini 4 Pro", "desc", None, None, None, None, 3,
            ]
        ]
    )
    body.append([16])  # tier flag → capacity 3
    body.append([])
    _mock_batch(c, "otAQ7b", body)
    registry = c.discover_models()
    assert registry["capacity"] == 3
    assert registry["models"][0]["model_id"] == "e6fa609c3fa255c0"
    assert registry["models"][0]["model_number"] == 3
    assert "pro" in registry["models"][0]["name"]
    # Cached: the second call returns the same dict without another RPC.
    assert c.discover_models() is registry


def test_discover_models_refresh_bypasses_cache():
    c = _client()
    body1 = [None] * 15
    body1.append([])
    body1.append([])
    body1.append([])
    calls = {"n": 0}

    def fake_batch(rpcs, source_path="/app"):
        calls["n"] += 1
        return _frames("otAQ7b", body1)

    c.batch_execute = fake_batch
    c.discover_models()
    c.discover_models()
    assert calls["n"] == 1
    c.discover_models(refresh=True)
    assert calls["n"] == 2
