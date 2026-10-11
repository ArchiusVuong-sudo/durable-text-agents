"""Live provider: one LiveKit LLM call = one Claude Agent SDK query.

Tools are exposed to the model as in-process SDK MCP tools, but a PreToolUse
hook answers ``defer`` for every call, so the SDK never runs a handler. The run
ends with ``ResultMessage.deferred_tool_use``, which we hand back to LiveKit as
a ``FunctionToolCall``. LiveKit then runs the real ``@function_tool`` and asks
for the next step, which keeps the tool loop, handoffs and RunResult in the
framework.

Isolation on every query: no built-in tools, only this call's MCP tools in the
allowlist, ``dontAsk`` permissions (anything not pre-approved is denied), no
settings files, strict MCP config, no session persistence, a fresh temp cwd and
a bounded number of turns.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import random
import shutil
import signal
import tempfile
import time
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    HookContext,
    HookInput,
    HookJSONOutput,
    HookMatcher,
    ResultMessage,
    TextBlock,
    create_sdk_mcp_server,
    query,
    tool,
)
from claude_agent_sdk._errors import ClaudeSDKError, ProcessError
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

from ..envsafe import child_env
from .base import SMS_RULES, CompletionLLM, LLMCall, LLMReply, LLMRequest, ToolCallReply

log = logging.getLogger(__name__)

SERVER = "lk"
PREFIX = f"mcp__{SERVER}__"
RETRY_MARKERS = ("overloaded", "rate limit", "rate_limit", "529", "503", "502", "500", "server_error")


class RetryableLLMError(Exception):
    pass


class _TrackedTransport(SubprocessCLITransport):
    """Remembers the CLI child pid so a cancelled call can end it for sure."""

    pid: int | None = None

    async def connect(self) -> None:
        await super().connect()
        proc = getattr(self, "_process", None)
        self.pid = getattr(proc, "pid", None)


def _kill_child(pid: int | None) -> None:
    if not pid:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(pid, signal.SIGKILL)


async def _defer_hook(input_data: HookInput, tool_use_id: str | None, context: HookContext) -> HookJSONOutput:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "defer",
            "permissionDecisionReason": "the host runtime executes tools",
        }
    }


def _is_retryable(text: str, status: int | None = None) -> bool:
    if status is not None and (status == 429 or status >= 500):
        return True
    low = text.lower()
    return any(m in low for m in RETRY_MARKERS)


class ClaudeSDKLLM(CompletionLLM):
    def __init__(
        self,
        model: str | None = None,
        *,
        max_attempts: int = 5,
        base_delay_s: float = 1.5,
        effort: str = "low",
    ) -> None:
        super().__init__(model or "claude-sonnet-5-5")
        self._max_attempts = max_attempts
        self._base_delay = base_delay_s
        self._effort = effort

    def _options(self, req: LLMRequest, cwd: str) -> ClaudeAgentOptions:
        sdk_tools = []
        for spec in req.tools:

            async def _never(args: dict[str, Any]) -> dict[str, Any]:  # deferred, never runs
                return {"content": [{"type": "text", "text": "deferred to host"}]}

            sdk_tools.append(tool(spec.name, spec.description, spec.input_schema)(_never))
        mcp = (
            {SERVER: create_sdk_mcp_server(name=SERVER, version="0.1.0", tools=sdk_tools)}
            if sdk_tools
            else {}
        )
        hooks = (
            {"PreToolUse": [HookMatcher(matcher=PREFIX + ".*", hooks=[_defer_hook])]} if sdk_tools else None
        )
        return ClaudeAgentOptions(
            system_prompt=f"{req.system}\n\n{SMS_RULES}",
            tools=[],
            allowed_tools=[PREFIX + t.name for t in req.tools],
            mcp_servers=mcp,  # type: ignore[arg-type]
            strict_mcp_config=True,
            hooks=hooks,  # type: ignore[arg-type]
            permission_mode="dontAsk",
            setting_sources=[],
            cwd=cwd,
            max_turns=2,
            model=self._model_name,
            effort=self._effort,  # type: ignore[arg-type]
            env=child_env(),
            extra_args={"no-session-persistence": None},
        )

    async def complete(self, req: LLMRequest) -> LLMCall:
        started = time.monotonic()
        last_err: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                call = await self._once(req)
                call.attempts = attempt
                call.duration_ms = int((time.monotonic() - started) * 1000)
                return call
            except RetryableLLMError as exc:
                last_err = exc
            except ProcessError as exc:
                text = f"{exc} {getattr(exc, 'stderr', '') or ''}"
                status = getattr(exc, "api_error_status", None)
                if not _is_retryable(text, status):
                    raise
                last_err = exc
            if attempt < self._max_attempts:
                delay = self._base_delay * (2 ** (attempt - 1)) * (0.75 + random.random() / 2)
                log.warning("LLM call retry %s/%s in %.1fs: %s", attempt, self._max_attempts, delay, last_err)
                await asyncio.sleep(delay)
        raise ClaudeSDKError(f"LLM call failed after {self._max_attempts} attempts: {last_err}")

    async def _once(self, req: LLMRequest) -> LLMCall:
        cwd = tempfile.mkdtemp(prefix="dta-llm-")
        opts = self._options(req, cwd)
        prompt = (
            "Conversation so far:\n"
            + "\n".join(req.transcript)
            + "\n\nContinue as the assistant with the next single step."
        )
        transport = _TrackedTransport(prompt=prompt, options=opts)
        texts: list[str] = []
        result: ResultMessage | None = None
        model = self._model_name
        try:
            async for msg in query(prompt=prompt, options=opts, transport=transport):
                if isinstance(msg, AssistantMessage):
                    model = msg.model or model
                    if msg.error in ("rate_limit", "server_error"):
                        raise RetryableLLMError(f"assistant error: {msg.error}")
                    texts.extend(b.text for b in msg.content if isinstance(b, TextBlock) and b.text.strip())
                elif isinstance(msg, ResultMessage):
                    result = msg
        except asyncio.CancelledError:
            _kill_child(transport.pid)  # a superseded turn must not leave a CLI child behind
            raise
        finally:
            shutil.rmtree(cwd, ignore_errors=True)
        if result is None:
            raise RetryableLLMError("no result message from the CLI")
        if result.is_error and result.deferred_tool_use is None:
            detail = f"{result.subtype} {result.result or ''} {result.errors or ''}"
            if _is_retryable(detail, result.api_error_status):
                raise RetryableLLMError(detail)
            raise ClaudeSDKError(f"LLM error: {detail}")
        usage = result.usage or {}
        reply = LLMReply()
        deferred = result.deferred_tool_use
        if deferred is not None:
            reply.tool_call = ToolCallReply(
                name=deferred.name.removeprefix(PREFIX),
                arguments=dict(deferred.input or {}),
                call_id=deferred.id,
            )
        else:
            reply.text = "\n".join(texts).strip() or (result.result or "").strip()
        return LLMCall(
            key=req.key,
            agent_id=req.agent_id,
            source="live",
            model=model,
            request_tail=req.tail,
            tools=[t.name for t in req.tools],
            reply=reply,
            duration_ms=result.duration_ms,
            api_ms=result.duration_api_ms,
            input_tokens=_tokens_in(usage),
            output_tokens=usage.get("output_tokens"),
            cost_usd=result.total_cost_usd,
        )


def _tokens_in(usage: dict[str, Any]) -> int | None:
    if not usage:
        return None
    return int(
        usage.get("input_tokens", 0)
        + usage.get("cache_read_input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0)
    )
