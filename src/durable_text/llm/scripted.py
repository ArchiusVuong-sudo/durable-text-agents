"""A deterministic policy LLM for tests and as the labelled replay fallback.

It reads the same ``LLMRequest`` a live model gets and picks the next step with
simple rules per agent. It is not meant to be clever, only predictable.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import CompletionLLM, LLMCall, LLMReply, LLMRequest, ToolCallReply

_CALLED = re.compile(r"^\[assistant called (\w+)\] (.*)$")
_RESULT = "[tool result] "


class ScriptedLLM(CompletionLLM):
    def __init__(self, source: str = "tape") -> None:
        super().__init__("scripted-policy")
        self._source = source
        self._n = 0

    async def complete(self, req: LLMRequest) -> LLMCall:
        reply = decide(req)
        if reply.tool_call is not None:
            self._n += 1
            reply.tool_call.call_id = f"call_{req.key[:10]}_{self._n}"
        return LLMCall(
            key=req.key,
            agent_id=req.agent_id,
            source=self._source,  # type: ignore[arg-type]
            model=self.model,
            request_tail=req.tail,
            tools=[t.name for t in req.tools],
            reply=reply,
            duration_ms=0,
        )


def _tool(name: str, **args: Any) -> LLMReply:
    return LLMReply(tool_call=ToolCallReply(name=name, arguments=args, call_id=""))


def _text(text: str) -> LLMReply:
    return LLMReply(text=text)


def _last_step(lines: list[str]) -> tuple[str, str, dict[str, Any] | str]:
    """Return ('user', text, '') or ('tool', name, parsed_result) for the newest step."""
    last_call = ""
    for line in reversed(lines):
        m = _CALLED.match(line)
        if m:
            last_call = m.group(1)
            break
    tail = lines[-1] if lines else ""
    if tail.startswith(_RESULT):
        raw = tail[len(_RESULT):]
        try:
            return "tool", last_call, json.loads(raw)
        except json.JSONDecodeError:
            return "tool", last_call, raw
    return "user", tail.split("] ", 1)[-1], ""


def _latest_user(lines: list[str]) -> str:
    for line in reversed(lines):
        if line.startswith("[user] "):
            return line[len("[user] "):]
    return ""


def decide(req: LLMRequest) -> LLMReply:
    kind, what, result = _last_step(req.transcript)
    if req.agent_id == "front_desk":
        if kind == "user" and "transfer_to_scheduler" in [t.name for t in req.tools]:
            return _tool("transfer_to_scheduler", issue=_latest_user(req.transcript)[:120])
        return _text("Thanks for texting Kettle & Coil Repair. What needs fixing?")
    if req.agent_id == "scheduler":
        return _scheduler(kind, what, result, req)
    if req.agent_id == "dispatch":
        return _dispatch(kind, what, result, req)
    return _text("Thanks, we got your message.")


def _scheduler(kind: str, what: str, result: Any, req: LLMRequest) -> LLMReply:
    text = _latest_user(req.transcript).lower()
    if kind == "user" and "<instructions>" in "\n".join(req.transcript[-3:]):
        event = re.search(r"Event from [^:]+: ([^.]+\.)", "\n".join(req.transcript[-3:]))
        return _text(f"Kettle & Coil update: {event.group(1) if event else 'thanks for waiting.'}")
    if kind == "tool":
        if what == "transfer_to_scheduler":
            return _tool("list_slots", day="tomorrow")
        if what == "list_slots" and isinstance(result, dict):
            pick = _pick_slot(result, text)
            if pick and any(w in text for w in ("works", "better", "book", "morning", "afternoon")) \
                    and "dishwasher" not in text:
                return _tool("hold_slot", slot_id=pick)
            windows = ", ".join(s["window"].split(" ", 3)[-1] for s in result.get("open", []))
            return _text(f"Sorry about that. Tomorrow we have {windows}. Which works for you?")
        if what == "hold_slot":
            return _tool("request_dispatch", notes="see issue summary")
        if what == "request_dispatch":
            return _text(
                "Done, that time is held. Dispatch is confirming a technician and I will text "
                "you once it is confirmed."
            )
        if what == "check_booking" and isinstance(result, dict):
            return _text(f"Your visit {result.get('held_window')} is held: {result.get('dispatch')}.")
        return _text("Thanks, noted.")
    if any(w in text for w in ("update", "status", "any news")):
        return _tool("check_booking")
    return _tool("list_slots", day="tomorrow")


def _pick_slot(result: dict[str, Any], text: str) -> str | None:
    slots = result.get("open", [])
    if not slots:
        return None
    if "afternoon" in text:
        for s in slots:
            if int(s["slot_id"].split("-")[1][:2]) >= 12:
                return str(s["slot_id"])
    return str(slots[0]["slot_id"])


def _dispatch(kind: str, what: str, result: Any, req: LLMRequest) -> LLMReply:
    if kind == "tool" and what == "list_technicians" and isinstance(result, dict):
        for t in result.get("technicians", []):
            if t.get("in_zone") and t.get("free_for_slot"):
                return _tool(
                    "propose_assignment",
                    technician_id=t["technician_id"],
                    reason=f"{t['name']} has the skill, works in the zone and is free.",
                )
        return _text("No technician fits.")
    task = re.search(r"Task: (\{.*\})", req.system)
    data = json.loads(task.group(1)) if task else {}
    return _tool("list_technicians", skill=data.get("skill", "dishwasher"), zone=data.get("zone", "East"))
