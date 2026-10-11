"""Environment allowlist for every process that hosts the Claude Agent SDK.

The SDK merges ``ClaudeAgentOptions.env`` over ``os.environ``, so inherited
variables cannot be removed through options. Instead the supervisor starts the
worker with ``child_env()`` and the worker scrubs its own ``os.environ`` at
start, so the Claude Code CLI child only ever sees the allowlist.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

ALLOWED_KEYS: tuple[str, ...] = ("PATH", "HOME", "USER", "LANG", "TMPDIR", "ANTHROPIC_API_KEY")


def child_env(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return only the allowlisted variables that are set and non-empty."""
    src = os.environ if source is None else source
    env = {k: src[k] for k in ALLOWED_KEYS if src.get(k)}
    return {k: v for k, v in env.items() if not _is_blocked(k)}


def _is_blocked(key: str) -> bool:
    return key == "CLAUDECODE" or key.startswith("CLAUDE_CODE_")


def scrub_process_env() -> list[str]:
    """Reduce ``os.environ`` to the allowlist. Returns the names that were removed."""
    keep = child_env()
    removed = sorted(k for k in os.environ if k not in keep)
    os.environ.clear()
    os.environ.update(keep)
    return removed
