"""Pick the provider: live when the SDK can run, replay otherwise."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..config import Settings
from ..envsafe import child_env
from .base import CompletionLLM

Mode = Literal["live", "replay", "scripted"]


@dataclass(frozen=True)
class ModeInfo:
    mode: Mode
    reason: str


def bundled_cli() -> Path | None:
    try:
        import claude_agent_sdk

        path = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
        if path.exists():
            return path
    except ImportError:
        return None
    found = shutil.which("claude")
    return Path(found) if found else None


def detect_mode(settings: Settings) -> ModeInfo:
    if settings.replay_forced:
        return ModeInfo("replay", "DEMO_REPLAY=1")
    env = child_env()
    if env.get("ANTHROPIC_API_KEY"):
        return ModeInfo("live", "ANTHROPIC_API_KEY is set")
    cli = bundled_cli()
    if cli is None:
        return ModeInfo("replay", "no Claude Code CLI found")
    try:
        out = subprocess.run(
            [str(cli), "auth", "status"],
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
            cwd=env.get("TMPDIR") or "/tmp",
        )
        status = json.loads(out.stdout or "{}")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return ModeInfo("replay", "could not read the Claude Code login state")
    if status.get("loggedIn"):
        return ModeInfo("live", "local Claude Code login")
    return ModeInfo("replay", "no Claude Code login and no ANTHROPIC_API_KEY")


def make_llm(mode: Mode, settings: Settings) -> CompletionLLM:
    if mode == "live":
        from .claude_sdk import ClaudeSDKLLM

        live: CompletionLLM = ClaudeSDKLLM(settings.model)
        if settings.record_path:
            from .replay import RecordingLLM

            live = RecordingLLM(live, settings.record_path)
        return live
    if mode == "replay":
        from .replay import ReplayLLM

        return ReplayLLM(settings.fixtures, scale=settings.replay_scale)
    from .scripted import ScriptedLLM

    return ScriptedLLM()
