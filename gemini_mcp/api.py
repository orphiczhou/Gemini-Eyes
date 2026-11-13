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
import os
import time
import uuid
from typing import Any, Optional, Union

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from . import usage
from .gemini_client import (
    DEFAULT_MODEL,
    DEFAULT_THINKING,
    GeminiWebError,
    MODEL_ALIASES,
    MODEL_CATALOG,
    THINKING_LEVELS,
    _normalise_model_name,
)
from .server import (
    _generate_serialized,
    _get_client,
    _metadata_for,
    _status_payload,
    _usage_payload,
)

_CREATED = int(time.time())
_VERSION = "0.1.0-api"

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


def _gemini_error_response(exc: GeminiWebError) -> JSONResponse:
    """Translate GeminiWebError into a terse HTTP error (never echoing upstream text)."""
    if exc.transport:
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
    frame, so the standardized reasoning_effort field never maps to 3 — it
    must be requested explicitly via gemini_thinking.
    """
    thinking = body.get("gemini_thinking")
    if thinking is not None:
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
    return JSONResponse(await run_in_threadpool(_healthz_payload))


async def _models(request: Request) -> JSONResponse:
    data = []
    for name, (model_id, capacity, number) in MODEL_CATALOG.items():
        data.append(
            {
                "id": f"gemini-{name}",
                "object": "model",
                "created": _CREATED,
                "owned_by": "gemini-web",
                "gemini": {"capacity": capacity, "mode_number": number},
            }
        )
    data.append(
        {
            "id": "gemini-legacy",
            "object": "model",
            "created": _CREATED,
            "owned_by": "gemini-web",
            "gemini": {"note": "model-less legacy payload (web default behaviour)"},
        }
    )
    return JSONResponse(
        {
            "object": "list",
            "data": data,
            "gemini": {
                "default_model": f"gemini-{DEFAULT_MODEL}",
                "default_thinking": DEFAULT_THINKING,
                "thinking_levels": dict(THINKING_LEVELS),
                "model_aliases": dict(MODEL_ALIASES),
            },
        }
    )


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
    if body.get("stream"):
        return _json_error(
            501,
            "streaming_unsupported",
            'SSE streaming is not implemented in this release; send "stream": false.',
        )

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
        model_name = key

    conv_id = body.get("gemini_conversation_id")
    resp_id = body.get("gemini_response_id")
    thinking = _resolve_thinking(body)

    prompt = _messages_to_prompt(messages, continue_conversation=bool(conv_id))
    if not prompt.strip():
        return _json_error(400, "invalid_request", "no user message content found in 'messages'")

    meta = _metadata_for(conv_id, resp_id) if conv_id else None
    try:
        result = await run_in_threadpool(
            _generate_serialized, prompt, meta, None, model_name, thinking
        )
    except GeminiWebError as exc:
        usage.record_error()
        return _gemini_error_response(exc)
    except Exception:  # noqa: BLE001 - never leak internals to the client
        usage.record_error()
        return _json_error(500, "internal_error", "unexpected server failure")

    # REST path counts its own traffic here; _generate_serialized stays
    # count-free so the MCP tool path (which counts in gemini_send_message)
    # and this path each report exactly once.
    usage.record_request(chars_in=len(prompt), chars_out=len(result.text))
    return JSONResponse(_chat_shell(str(requested_model or f"gemini-{DEFAULT_MODEL}"), prompt, result))


def _chat_shell(requested_model: str, prompt: str, result: Any) -> dict[str, Any]:
    """OpenAI chat.completion shell + a `gemini` extension object."""
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": result.text},
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
