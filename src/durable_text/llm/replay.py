"""Replay provider: answers from responses recorded on real live runs.

``fixtures/replay.json`` maps a request key (hash of system prompt, transcript
and tool names) to the recorded ``LLMCall``. A hit sleeps the recorded latency
(scaled by ``DTA_REPLAY_SCALE``) so timing-dependent behaviour, like a new text
superseding a running turn, still shows. A miss falls back to the scripted
policy and is labelled ``replay-fallback`` on the turn card.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
from pathlib import Path

from .base import CompletionLLM, LLMCall, LLMRequest
from .scripted import ScriptedLLM

FALLBACK_DELAY_S = 3.0
_BACKGROUND: set[asyncio.Future[LLMCall]] = set()  # recordings still finishing after a cancel


def load_fixtures(path: Path) -> dict[str, LLMCall]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {k: LLMCall.model_validate(v) for k, v in data.get("calls", {}).items()}


class ReplayLLM(CompletionLLM):
    def __init__(self, fixtures: Path, *, scale: float = 1.0) -> None:
        self._fixtures = load_fixtures(fixtures)
        super().__init__(self._recorded_model())
        self._scale = scale
        self._fallback = ScriptedLLM(source="replay-fallback")

    def _recorded_model(self) -> str:
        models = {c.model for c in self._fixtures.values()}
        return next(iter(models)) if len(models) == 1 else "recorded"

    @property
    def size(self) -> int:
        return len(self._fixtures)

    async def complete(self, req: LLMRequest) -> LLMCall:
        hit = self._fixtures.get(req.key)
        if hit is None:
            # Not recorded (the flow went another way): answer from the scripted policy,
            # labelled as such, after a short model-like pause so cancellation still works.
            if self._scale > 0:
                await asyncio.sleep(max(0.8, FALLBACK_DELAY_S * self._scale))
            return await self._fallback.complete(req)
        call = hit.model_copy(deep=True)
        if self._scale > 0:
            await asyncio.sleep(hit.duration_ms / 1000 * self._scale)
        call.source = "replay"
        call.recorded_ms = hit.duration_ms
        call.duration_ms = int(hit.duration_ms * self._scale)
        return call


class RecordingLLM(CompletionLLM):
    """Wraps a live provider and appends every call to a JSONL file."""

    def __init__(self, inner: CompletionLLM, path: Path) -> None:
        super().__init__(inner.model)
        self._inner = inner
        self._path = path

    async def complete(self, req: LLMRequest) -> LLMCall:
        # Shielded: when a turn is superseded mid-call, the call still finishes in the
        # background and is recorded, so replay can show the same cancelled turn.
        task = asyncio.ensure_future(self._record(req))
        _BACKGROUND.add(task)
        task.add_done_callback(_BACKGROUND.discard)
        return await asyncio.shield(task)

    async def _record(self, req: LLMRequest) -> LLMCall:
        call = await self._inner.complete(req)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write(json.dumps({"key": req.key, "call": call.model_dump()}) + "\n")
        return call


def merge_recording(jsonl: Path, out: Path) -> int:
    """Turn a recording into ``replay.json`` (first answer per key wins)."""
    calls: dict[str, dict[str, object]] = {}
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        calls.setdefault(item["key"], item["call"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"version": 1, "calls": calls}, indent=1, sort_keys=True) + "\n")
    return len(calls)
