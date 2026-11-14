"""gemini-web-mcp: MCP server for talking to gemini.google.com via its web interface.

No official API key is needed. The server "decompiles" (extracts and decrypts)
cookies from a local Chrome/Edge profile and replays them against Gemini's
internal RPC endpoints, exactly like the browser does.
"""

try:  # installed dist (uv editable install) — stays in sync with pyproject
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("gemini-web-mcp")
except Exception:  # noqa: BLE001 - not installed (e.g. bare source tree)
    __version__ = "0.2.0"  # keep in sync with pyproject.toml [project]
