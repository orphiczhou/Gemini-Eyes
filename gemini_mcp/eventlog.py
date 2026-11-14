"""Sanitized daemon event log (rotating file, metadata only).

One INFO-level record per observable daemon event: request start/end (model,
duration, chars_in/chars_out LENGTHS), conversation-chain hit/miss
(conversation_id, fingerprint prefix), model degradations and errors (error
code, HTTP status). The rotating file handler is attached to the package
logger ``gemini_mcp``, so events raised by gemini_client (model fallbacks)
land in the same file as the REST-layer events.

RED LINE — sanitization contract (enforced by convention at every call site):

* cookies, tokens and credentials are NEVER logged;
* prompt bodies and reply bodies are NEVER logged — only character counts;
* ids (conversation_id, fingerprint prefixes) are safe to log.

Path resolution: ``GEMINI_LOG_FILE`` env override wins; the default is
``<repo root>/.logs/gemini-mcp.log`` (repo root = parent of this package).
The directory is created on demand; rotation is 1 MB x 3 backups.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_PACKAGE_LOGGER = "gemini_mcp"
_MAX_BYTES = 1024 * 1024
_BACKUPS = 3

_last_path: Optional[Path] = None


def event_log_path() -> Path:
    """Resolve the log file location (env override > repo-root default)."""
    override = os.environ.get("GEMINI_LOG_FILE")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / ".logs" / "gemini-mcp.log"


def reset_event_logging() -> None:
    """Detach (and close) the rotating file handler; forget the path.

    Used by tests and by reconfiguration so handlers never leak across
    configurations or keep deleted temp files open on Windows.
    """
    global _last_path
    root = logging.getLogger(_PACKAGE_LOGGER)
    for handler in [h for h in root.handlers if isinstance(h, RotatingFileHandler)]:
        root.removeHandler(handler)
        handler.close()
    _last_path = None


def configure_event_logging() -> Optional[Path]:
    """(Re)attach the rotating file handler to the package logger.

    Idempotent: an existing rotating handler from an earlier configuration is
    replaced. Returns the resolved path, or None when the file cannot be
    opened (daemon keeps running, just without the event log).
    """
    global _last_path
    reset_event_logging()
    path = event_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path, maxBytes=_MAX_BYTES, backupCount=_BACKUPS, encoding="utf-8"
        )
    except OSError:
        return None
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger(_PACKAGE_LOGGER)
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    _last_path = path
    return path


def event_logger() -> logging.Logger:
    """Logger for one sanitized event line; configures lazily on first use.

    INFO records do not reach the console unless the process runs with
    --verbose (root logger at DEBUG/INFO); the rotating file is the sink.
    """
    if _last_path is None:
        configure_event_logging()
    return logging.getLogger(_PACKAGE_LOGGER)
