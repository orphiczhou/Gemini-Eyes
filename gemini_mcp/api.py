"""OpenAI-compatible REST facade over the GeminiWebClient (HTTP transport).

Mounted by server.py next to the MCP streamable app on the same port; every
non-MCP path lands here. Endpoints (all JSON):

    GET  /                     endpoint index (no auth)
    GET  /healthz              control-plane readiness probe (no auth)
    GET  /v1/models            selectable tiers, OpenAI list format
    GET  /v1/status            gemini_status passthrough
    GET  /v1/usage             local usage counters + fallback events
    POST /v1/chat/completions  chat (non-streaming MVP)

Auth mirrors the MCP side: ``Authorization: Bearer <token>`` with a
constant-time comparison; ``/healthz`` and ``/`` are exempt so the plugin
control plane and humans can probe without the token. ``token=None``
(--http-allow-anonymous) disables the check entirely, matching the MCP
middleware.

Design rules carried over from the feasibility review:

* generate() calls are serialized behind the process-wide lock
  (single-cookie single-writer, review P1-1) and pushed onto the threadpool
  so the asyncio loop serving SSE stays responsive.
* Error bodies never echo upstream response text (review P1-5) — only the
  mapped, human-phrased message plus the numeric server code.
* The web backend has no per-account usage API; usage reporting is local
  accounting (usage.py) and model-fallback events (1037/1050/1052).
"""

from __future__ import annotations

import hmac
import json
import os
import re
import time
import uuid
from typing import Any, Optional, Union

from starlette.applications import Starlette
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from . import usage
from .eventlog import event_logger
from .gemini_client import (
    DEFAULT_MODEL,
    DEFAULT_THINKING,
    DeepThinkUnavailable,
    GeminiWebError,
    MODEL_ALIASES,
    MODEL_CATALOG,
    THINKING_LEVELS,
    UNAVAILABLE_MODELS,
    _normalise_model_name,
    resolve_thinking_level,
)
from .server import (
    _generate_serialized,
    _get_client,
    _metadata_for,
    _status_payload,
    _usage_payload,
    conv_lookup,
    conv_register,
    session_key,
)

_CREATED = int(time.time())
_VERSION = "0.2.0-api"

# The web app decorates replies with UI-only follow-up suggestions that
# OpenAI-compatible clients would read as reply body — strip them (safe: the
# tag is self-contained and never spans lines).
_FOLLOWUP_RE = re.compile(r"\n*<FollowUp [^>]*/>\s*", re.IGNORECASE)

# GeminiWebError code -> (HTTP status, error.code string, human message).
# Deliberately terse: no upstream response fragments are ever echoed (P1-5).
_ERROR_MAP: dict[int, tuple[int, str, str]] = {
    1013: (503, "upstream_transient", "Upstream temporary error; retry shortly."),
    1037: (
        429,
        "upstream_rate_limited",
        "Usage limit exceeded for this model/account on the web plan; "
        "wait a bit or pick a lower tier (see GET /v1/models).",
    ),
    1050: (
        409,
        "conversation_model_conflict",
        "This conversation carries a different model; continue without a "
        "model override or start a new conversation.",
    ),
    1052: (400, "model_unavailable", "Requested model is invalid or unavailable for this account."),
    1060: (429, "ip_blocked", "IP temporarily blocked by Google; wait or change the outbound proxy."),
}


def _json_error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": "invalid_request_error", "code": code}},
        status_code=status,
    )


def _log_request_error(model: str, resp: JSONResponse) -> None:
    """Event-log one failed request: error code + HTTP status only (sanitized)."""
    code = "?"
    try:
        code = json.loads(resp.body.decode("utf-8")).get("error", {}).get("code", "?")
    except Exception:  # noqa: BLE001 - logging must never raise
        pass
    event_logger().info(
        "request_error model=%s code=%s http_status=%d", model, code, resp.status_code
    )


def _gemini_error_response(exc: GeminiWebError) -> JSONResponse:
    """Translate GeminiWebError into a terse HTTP error (never echoing upstream text)."""
    if getattr(exc, "stall", False):
        status, code, message = (
            504,
            "upstream_stalled",
            "Upstream produced no output for a long time (stalled). The model "
            "may be unavailable on this plan (e.g. Carbon/Ultra-only); try "
            "gemini-argon or gemini-pro (see GET /v1/models).",
        )
    elif exc.transport:
        status, code, message = 502, "upstream_transport", "Upstream connection failed (timeout/reset)."
    elif exc.code in _ERROR_MAP:
        status, code, message = _ERROR_MAP[exc.code]
    elif exc.code is None:
        # Token bootstrap failures (stale cookies) carry no numeric code.
        status, code, message = (
            503,
            "auth_session_expired",
            "Session cookies look stale; run gemini_status / GET /v1/status to diagnose.",
        )
    else:
        status, code, message = 502, "upstream_error", f"Upstream rejected the request (server code {exc.code})."
    return JSONResponse(
        {"error": {"message": message, "type": "gemini_web_error", "code": code}},
        status_code=status,
    )


class _ApiAuthMiddleware:
    """Bearer gate for the REST layer; exempt paths stay open (healthz/index)."""

    def __init__(self, app: Any, token: Optional[str]) -> None:
        self.app = app
        self._expected = b"Bearer " + token.encode() if token else None
        self._exempt = {"/healthz", "/"}

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") == "http" and self._expected is not None:
            if scope.get("path", "") not in self._exempt:
                headers = dict(scope.get("headers") or [])
                if not hmac.compare_digest(headers.get(b"authorization", b""), self._expected):
                    body = b'{"error": {"message": "Authentication required", "code": "unauthorized"}}'
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 401,
                            "headers": [
                                (b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode()),
                                (b"www-authenticate", b"Bearer"),
                            ],
                        }
                    )
                    await send({"type": "http.response.body", "body": body})
                    return
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def _content_to_text(content: Any) -> str:
    """Flatten an OpenAI message content (string or parts array) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return ""


def _messages_to_prompt(messages: list[dict[str, Any]], continue_conversation: bool) -> str:
    """Build the single web prompt from an OpenAI messages array.

    * role=system items become an <system instructions> prefix — the web UI
      has no dedicated system slot, so this is prefix injection (best effort,
      disclosed to the caller in the README).
    * Continuing a server-side conversation (``continue_conversation``) only
      sends the last user turn — the earlier turns already live in the web
      conversation. A fresh conversation renders the whole non-system history
      as a labelled transcript so the model still sees the context.
    """
    prefix = ""
    turns: list[tuple[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user"))
        text = _content_to_text(message.get("content"))
        if not text:
            continue
        if role == "system":
            prefix += f"<system instructions>{text}</system instructions>\n\n"
        else:
            turns.append((role, text))
    if not turns:
        return ""
    if continue_conversation:
        last_user = next((t for t in reversed(turns) if t[0] == "user"), None)
        return prefix + (last_user[1] if last_user else turns[-1][1])
    if len(turns) == 1:
        return prefix + turns[0][1]
    body = "\n\n".join(
        f"[{role}]\n{text}" for role, text in turns
    )
    return prefix + body + "\n\n[assistant]"


def _healthz_payload() -> dict[str, Any]:
    """Light, network-free readiness signal for the plugin control plane."""
    out: dict[str, Any] = {
        "ok": True,
        "pid": os.getpid(),
        "version": _VERSION,
        "uptime_seconds": usage.snapshot()["uptime_seconds"],
        "ready": False,
        "initError": None,
    }
    try:
        _get_client()  # constructs the client from the cookie file (no network)
        out["ready"] = True
    except Exception as e:  # noqa: BLE001 - probe must not crash
        out["ok"] = False
        out["initError"] = str(e)
    return out


def _resolve_thinking(body: dict[str, Any]) -> Optional[Union[int, str]]:
    """Explicit gemini_thinking wins; OpenAI reasoning_effort maps onto 1/2.

    Level 3 (Deep Think) answers asynchronously with a placeholder first
    frame, so it is refused up front: ``gemini_thinking`` values that resolve
    to 3 (explicit int/"3" or Deep-Think spellings) raise
    :class:`DeepThinkUnavailable`, which ``_chat`` translates into 400
    ``deep_think_unavailable``. The standardized ``reasoning_effort`` field
    never maps to 3 (minimal/low -> 1, everything else -> 2).
    """
    thinking = body.get("gemini_thinking")
    if thinking is not None:
        resolve_thinking_level(thinking)  # validation only — raises on level 3
        return thinking
    effort = body.get("reasoning_effort")
    if isinstance(effort, str) and effort.strip():
        key = effort.strip().lower()
        if key in ("minimal", "low"):
            return 1
        return 2  # medium/high/default
    return None


# ---------------------------------------------------------------------------
# Route handlers (async shells; blocking work goes through run_in_threadpool)
# ---------------------------------------------------------------------------


async def _index(request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "service": "gemini-web-mcp REST facade",
            "version": _VERSION,
            "endpoints": [
                "GET /healthz",
                "GET /v1/models",
                "GET /v1/status",
                "GET /v1/usage",
                "POST /v1/chat/completions",
            ],
        }
    )


async def _healthz(request: Request) -> JSONResponse:
    # Deliberately NOT via run_in_threadpool: during an upstream stall the
    # threadpool can be saturated by hung streams, and the health probe is
    # the one endpoint that must stay reachable (it is memory-only once the
    # client is constructed).
    return JSONResponse(_healthz_payload())


def _models_payload() -> dict[str, Any]:
    data = []
    for name, (model_id, capacity, number) in MODEL_CATALOG.items():
        entry = {
            "id": f"gemini-{name}",
            "object": "model",
            "created": _CREATED,
            "owned_by": "gemini-web",
            "gemini": {"capacity": capacity, "mode_number": number},
        }
        if name in UNAVAILABLE_MODELS:
            # Plan blocklist: listed for roster transparency, never servable.
            entry["available"] = False
            entry["unavailable_reason"] = UNAVAILABLE_MODELS[name]
        data.append(entry)
    data.append(
        {
            "id": "gemini-legacy",
            "object": "model",
            "created": _CREATED,
            "owned_by": "gemini-web",
            "gemini": {"note": "model-less legacy payload (web default behaviour)"},
        }
    )
    out: dict[str, Any] = {
        "object": "list",
        "data": data,
        # Protocol capabilities this facade actually supports. The web backend
        # has no tool channel, so tool/function traffic is sanitized away
        # (see _sanitize_messages) instead of being half-honoured.
        "capabilities": {"function_calling": False},
        "gemini": {
            "default_model": f"gemini-{DEFAULT_MODEL}",
            "default_thinking": DEFAULT_THINKING,
            "thinking_levels": dict(THINKING_LEVELS),
            "model_aliases": dict(MODEL_ALIASES),
        },
    }
    out["gemini"]["thinking_guide"] = [
        {"label": "低", "value": 1, "api": "同步", "note": "最快，轻任务"},
        {"label": "中", "value": 2, "api": "同步", "note": "默认，质量/延迟平衡"},
        {"label": "高", "value": 2, "api": "同步", "note": "与中同级（扩展思考），可用"},
        {
            "label": "Deep Think",
            "value": 3,
            "api": "不可用",
            "note": "等级 3 为异步占位（首响应仅为处理中提示），本接口直接拒绝：deep_think_unavailable",
        },
    ]
    out["gemini"]["discovered_note"] = (
        "discovered.models 来自账号实时发现：name=界面模型名（如 4 Carbon），"
        "display_name=界面选项名（如 高难任务），二者均可作 model 参数"
    )
    # Live account discovery (otAQ7b): served from the client-side cache
    # warmed by the background prewarm thread at startup. NEVER discovered
    # inline here — the RPC needs the generate lock and the network, and a
    # cold request would hang on both (observed as a hung /v1/models).
    try:
        client = _get_client()
        out["gemini"]["discovered"] = client.model_registry_snapshot()
    except Exception:  # noqa: BLE001 - discovery is enhancement, not contract
        out["gemini"]["discovered"] = None
    return out


async def _models(request: Request) -> JSONResponse:
    return JSONResponse(await run_in_threadpool(_models_payload))


async def _status(request: Request) -> JSONResponse:
    return JSONResponse(await run_in_threadpool(_status_payload))


async def _usage(request: Request) -> JSONResponse:
    return JSONResponse(await run_in_threadpool(_usage_payload))


async def _chat(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - malformed body
        return _json_error(400, "invalid_json", "request body must be JSON")
    if not isinstance(body, dict):
        return _json_error(400, "invalid_request", "request body must be a JSON object")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return _json_error(400, "invalid_request", "'messages' must be a non-empty array")

    # Model: normalise (gemini- prefix / aliases) and validate explicitly —
    # silently swapping an unknown name for the top tier would mislead clients.
    model_name: Optional[str] = None
    requested_model = body.get("model")
    if requested_model:
        key = _normalise_model_name(str(requested_model))
        valid = set(MODEL_CATALOG) | {"legacy"} | set(MODEL_ALIASES.values())
        if key not in valid:
            return _json_error(
                400,
                "unknown_model",
                f"unknown model {requested_model!r}; see GET /v1/models",
            )
        # Plan-availability gate (Ultra-only tiers would stall the stream for
        # minutes on this plan — reject before any upstream traffic).
        if key in UNAVAILABLE_MODELS:
            return _json_error(
                400,
                "model_unavailable",
                f"{UNAVAILABLE_MODELS[key]}. Try gemini-argon or gemini-pro "
                "instead (see GET /v1/models).",
            )
        model_name = key

    conv_id = body.get("gemini_conversation_id")
    resp_id = body.get("gemini_response_id")
    try:
        thinking = _resolve_thinking(body)
    except DeepThinkUnavailable:
        return _json_error(
            400,
            "deep_think_unavailable",
            "Thinking level 3 = Deep Think is an asynchronous placeholder and is "
            "not available on this facade; use 低/中/高 (=1/2/2).",
        )

    texts = _messages_to_texts(messages)
    # Fingerprint scope: the normalized catalog name (no gemini- prefix), so
    # "gemini-argon" and an unset model hash identically and the chain stays
    # intact across clients that toggle the prefix. Legacy table entries with
    # the old prefixed keys simply age out via the 90-min TTL.
    model_key = model_name or DEFAULT_MODEL
    # Session resolution: explicit id wins; otherwise fingerprint the history
    # BEFORE the last user turn (the next request's parent key equals this
    # turn's child key — see server.session_key). Miss → fresh conversation.
    rec = None
    if conv_id:
        rec = conv_lookup(texts, model_key, conv_id)
    else:
        rec = conv_lookup(texts[:-1], model_key) if texts else None

    prompt = _messages_to_prompt(messages, continue_conversation=rec is not None)
    if not prompt.strip():
        return _json_error(400, "invalid_request", "no user message content found in 'messages'")

    if rec is not None:
        meta = list(rec.metadata)
        if conv_id and resp_id:
            meta[1] = resp_id
    else:
        meta = _metadata_for(conv_id, resp_id) if conv_id else None

    # Sanitized event log: ids, model labels and LENGTHS only — never prompt
    # or reply bodies (see eventlog.py red lines).
    ev = event_logger()
    started = time.time()
    fp_input = texts if conv_id else texts[:-1]
    fp8 = session_key(fp_input, model_key)[:8] if fp_input else "-"
    if rec is not None:
        hit_cid = str((rec.metadata or [""])[0] or "?")
        ev.info("session_hit conversation_id=%s fingerprint=%s", hit_cid, fp8)
    else:
        ev.info("session_miss fingerprint=%s", fp8)
    will_stream = body.get("stream") is True or body.get("stream") == "true"
    ev.info(
        "request_start model=%s stream=%s chars_in=%d", model_key, will_stream, len(prompt)
    )

    if will_stream:
        return _chat_stream(body, prompt, meta, model_name, thinking, texts, model_key, started)
    try:
        result = await run_in_threadpool(
            _generate_serialized, prompt, meta, None, model_name, thinking
        )
    except GeminiWebError as exc:
        usage.record_error()
        resp = _gemini_error_response(exc)
        _log_request_error(model_key, resp)
        return resp
    except Exception:  # noqa: BLE001 - never leak internals to the client
        usage.record_error()
        # Traceback frames only (no payload data) — the one branch where the
        # cause is unknown and operators need it in the event log.
        ev.exception("request_error model=%s code=internal_error http_status=500", model_key)
        return _json_error(500, "internal_error", "unexpected server failure")

    # REST path counts its own traffic here; _generate_serialized stays
    # count-free so the MCP tool path (which counts in gemini_send_message)
    # and this path each report exactly once.
    usage.record_request(chars_in=len(prompt), chars_out=len(result.text))
    ev.info(
        "request_end model=%s elapsed=%.2f chars_in=%d chars_out=%d conversation_id=%s",
        model_key,
        time.time() - started,
        len(prompt),
        len(result.text),
        result.conversation_id,
    )
    clean_text = _FOLLOWUP_RE.sub("", result.text).strip()
    conv_register(result, texts + [f"assistant: {clean_text}"], str(model_key))
    return JSONResponse(_chat_shell(str(requested_model or f"gemini-{DEFAULT_MODEL}"), prompt, result))


def _chat_stream(
    body: dict[str, Any],
    prompt: str,
    meta: Optional[list[Any]],
    model_name: Optional[str],
    thinking: Optional[Union[int, str]],
    texts: Optional[list[str]] = None,
    model_key: Optional[str] = None,
    started: Optional[float] = None,
) -> StreamingResponse:
    """SSE stream: role frame, content/thinking deltas, ping keep-alives when
    the upstream stalls, final frame with the gemini extension, then [DONE].
    Client disconnect cancels the upstream call at the next frame boundary.

    ``model_key`` is the fingerprint-scope model name from ``_chat`` — it must
    be threaded in explicitly (a bare module-global lookup here once raised
    NameError at the register step of every streamed turn). ``started`` is the
    request-start timestamp for the sanitized event log."""
    requested_model = str(body.get("model") or f"gemini-{DEFAULT_MODEL}")
    client = _get_client()
    cancel = __import__("threading").Event()
    gen = client.generate_stream(
        prompt, metadata=meta, model=model_name, thinking=thinking, cancel=cancel
    )
    chunk_id = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    sent_chars = {"out": 0}
    ev = event_logger()
    started = started if started is not None else float(created)

    def frame(delta: dict[str, Any], finish: Optional[str] = None, extra: Optional[dict[str, Any]] = None) -> str:
        payload: dict[str, Any] = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": requested_model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if extra:
            payload["gemini"] = extra
        return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    async def sse() -> Any:
        try:
            yield frame({"role": "assistant", "content": ""})
            async for item in iterate_in_threadpool(gen):
                if item["type"] == "ping":
                    yield ": ping\n\n"  # SSE comment: proxies and clients see liveness
                    continue
                if item["type"] == "delta":
                    delta: dict[str, Any] = {}
                    if item["text"]:
                        # The FollowUp tag arrives whole inside one snapshot
                        # frame, so scrubbing per-delta is safe (same rule as
                        # the non-streaming body cleaner).
                        text_delta = _FOLLOWUP_RE.sub("", item["text"])
                        if text_delta:
                            delta["content"] = text_delta
                            sent_chars["out"] += len(text_delta)
                    if item["thoughts"]:
                        delta["reasoning_content"] = item["thoughts"]
                    if delta:
                        yield frame(delta)
                else:
                    result = item["result"]
                    usage.record_request(chars_in=len(prompt), chars_out=len(result.text))
                    from .server import _remember

                    _remember(result)
                    ev.info(
                        "request_end model=%s stream=true elapsed=%.2f "
                        "chars_in=%d chars_out=%d conversation_id=%s",
                        model_key or "-",
                        time.time() - started,
                        len(prompt),
                        len(result.text),
                        result.conversation_id,
                    )
                    if texts is not None:
                        clean_text = _FOLLOWUP_RE.sub("", result.text).strip()
                        conv_register(result, texts + [f"assistant: {clean_text}"], str(model_key or DEFAULT_MODEL))
                    yield frame(
                        {},
                        "stop",
                        {
                            "conversation_id": result.conversation_id,
                            "response_id": result.response_id,
                            "candidate_id": result.candidate_id,
                            "served_model": result.model,
                            "completed": result.completed,
                        },
                    )
        except GeminiWebError as exc:
            usage.record_error()
            error = _gemini_error_response(exc)
            _log_request_error(model_key or "-", error)
            yield "data: " + json.dumps(json.loads(error.body.decode("utf-8")), ensure_ascii=False) + "\n\n"
        except Exception:  # noqa: BLE001 - never leak internals into the stream
            usage.record_error()
            ev.info(
                "request_error model=%s stream=true code=internal_error http_status=200",
                model_key or "-",
            )
            yield (
                'data: {"error": {"message": "unexpected server failure", '
                '"type": "internal_error", "code": "internal_error"}}\n\n'
            )
        finally:
            # Consumer gone (or stream finished): stop the upstream at the
            # next frame boundary instead of letting it hold the generate
            # lock for minutes burning quota on a dead connection.
            cancel.set()
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _messages_to_texts(messages: list[dict[str, Any]]) -> list[str]:
    """Flatten non-system messages into the ordered texts used for the
    conversation fingerprint (role-prefixed, stripped)."""
    texts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user"))
        if role == "system":
            continue
        text = _content_to_text(message.get("content")).strip()
        if text:
            texts.append(f"{role}: {text}")
    return texts


def _chat_shell(requested_model: str, prompt: str, result: Any) -> dict[str, Any]:
    """OpenAI chat.completion shell + a `gemini` extension object."""
    clean_text = _FOLLOWUP_RE.sub("", result.text).strip()
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": clean_text},
            }
        ],
        "usage": {
            "prompt_tokens": max(1, len(prompt) // 4),
            "completion_tokens": max(1, len(result.text) // 4),
            "total_tokens": max(2, (len(prompt) + len(result.text)) // 4),
            "note": "character-based estimate; the web backend has no token accounting",
        },
        "gemini": {
            "conversation_id": result.conversation_id,
            "response_id": result.response_id,
            "candidate_id": result.candidate_id,
            "served_model": result.model,
            "completed": result.completed,
            "thoughts": result.thoughts or None,
        },
    }


def build_api_app(token: Optional[str]) -> Any:
    """ASGI app for the REST facade; ``token`` gates everything but / and /healthz."""
    routes = [
        Route("/", _index, methods=["GET"]),
        Route("/healthz", _healthz, methods=["GET"]),
        Route("/v1/models", _models, methods=["GET"]),
        Route("/v1/status", _status, methods=["GET"]),
        Route("/v1/usage", _usage, methods=["GET"]),
        Route("/v1/chat/completions", _chat, methods=["POST"]),
    ]
    app = Starlette(routes=routes)
    return _ApiAuthMiddleware(app, token)
