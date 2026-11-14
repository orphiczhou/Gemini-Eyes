"""Synthetic tests for the streaming layer (delta diff + generate_stream)."""

import json

from gemini_mcp.gemini_client import (
    GeminiWebClient,
    GeminiWebError,
    get_clean_text,
    get_delta_by_fp_len,
)


def test_clean_text_flicker():
    assert get_clean_text("hello") == "hello"
    assert get_clean_text("hello\n```") == "hello"
    # trailing escaped markdown is volatile and stripped with its tail
    assert get_clean_text("hello \\*") == "hello "


def test_delta_simple_growth():
    d1, s1 = get_delta_by_fp_len("abc", "", False)
    assert (d1, s1) == ("abc", "abc")
    d2, s2 = get_delta_by_fp_len("abcdef", s1, False)
    assert d2 == "def" and s2 == "abcdef"


def test_delta_final_adopts_clean_text():
    d, s = get_delta_by_fp_len("text\n```", "text", True)
    assert s == "text\n```"  # final keeps everything
    assert d.endswith("```") or d == ""


def test_generate_stream_events(monkeypatch):
    """generate_stream turns on_frame snapshots into ordered delta events."""
    c = GeminiWebClient({"__Secure-1PSID": "x"})

    def make_inner(text: str, completed: bool) -> list:
        inner: list = [None] * 43
        inner[1] = ["c_1", "r_1"]
        inner[4] = [["rc_1", [text], None, None, None, None, None, [2 if completed else 0]]]
        inner[25] = "ctx" if completed else None
        inner[42] = "4 Hydrogen"
        return inner

    frames = []
    for text, completed in [("Hel", False), ("Hello", False), ("Hello world", True)]:
        frames.append(["wrf.fr", "StreamGenerate", json.dumps(make_inner(text, completed)), None])

    captured = {}

    def stub_generate(prompt, metadata=None, files=None, model=None, thinking=None, on_frame=None):
        captured["prompt"] = prompt
        from gemini_mcp.gemini_client import GenerationResult

        r = GenerationResult()
        for f in frames:
            c._consume_frame(f, r)
            if on_frame:
                on_frame(f)
        return r

    monkeypatch.setattr(c, "generate", stub_generate)
    events = list(c.generate_stream("hi"))
    deltas = [e for e in events if e["type"] == "delta"]
    final = [e for e in events if e["type"] == "final"]
    assert len(final) == 1 and final[0]["result"].text == "Hello world"
    assembled = "".join(e["text"] for e in deltas)
    assert assembled == "Hello world"
    assert captured["prompt"] == "hi"


def test_generate_stream_rest_frames_do_not_reset_baseline(monkeypatch):
    """The web stream alternates content frames with empty rest frames; the
    baseline must survive them (otherwise the next content frame re-emits
    the whole answer — observed as "reply duplicated twice")."""
    c = GeminiWebClient({"__Secure-1PSID": "x"})

    def make_inner(text: str, completed: bool, with_candidates: bool = True) -> list:
        inner: list = [None] * 43
        inner[1] = ["c_1", "r_1"]
        if with_candidates:
            inner[4] = [["rc_1", [text], None, None, None, None, None, [2 if completed else 0]]]
        inner[25] = "ctx" if completed else None
        inner[42] = "4 Hydrogen"
        return inner

    seq = [
        ["wrf.fr", "StreamGenerate", json.dumps(make_inner("Hel", False)), None],
        ["wrf.fr", "StreamGenerate", json.dumps(make_inner("", False, with_candidates=False)), None],
        ["wrf.fr", "StreamGenerate", json.dumps(make_inner("Hello world", True)), None],
        ["wrf.fr", "StreamGenerate", json.dumps(make_inner("", False, with_candidates=False)), None],
    ]

    def stub_generate(prompt, metadata=None, files=None, model=None, thinking=None, on_frame=None):
        from gemini_mcp.gemini_client import GenerationResult

        r = GenerationResult()
        for f in seq:
            c._consume_frame(f, r)
            if on_frame:
                on_frame(f)
        return r

    monkeypatch.setattr(c, "generate", stub_generate)
    events = list(c.generate_stream("hi"))
    deltas = [e for e in events if e["type"] == "delta"]
    assembled = "".join(e["text"] for e in deltas)
    assert assembled == "Hello world", repr(assembled)
    assert "Hello worldHello world" not in assembled


def test_generate_stream_propagates_error(monkeypatch):
    c = GeminiWebClient({"__Secure-1PSID": "x"})

    def stub_generate(prompt, **kwargs):
        raise GeminiWebError("boom", code=1013)

    monkeypatch.setattr(c, "generate", stub_generate)
    try:
        list(c.generate_stream("hi"))
        raised = False
    except GeminiWebError as exc:
        raised = exc.code == 1013
    assert raised
