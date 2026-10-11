"""Provider interface: a LiveKit ``llm.LLM`` that makes one completion per call.

Every provider turns a LiveKit ``ChatContext`` plus tools into an ``LLMRequest``
and answers with an ``LLMReply`` that is either plain text or exactly one tool
call. The stream maps the reply back to LiveKit chunks, so the rest of the
framework (tool execution, handoffs, RunResult) runs unchanged.

Each call is logged as an ``LLMCall`` (latency, tokens, cost, source) so the
runtime can show per-turn metrics and the recorder can write replay fixtures.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from abc import abstractmethod
from typing import Any, Literal

from livekit.agents import APIConnectionError, llm
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, NotGivenOr
from pydantic import BaseModel, Field

AGENT_TAG = re.compile(r'<agent id="([a-z_]+)"')

SMS_RULES = (
    "You are replying over SMS. Write plain text only: no markdown, no lists, no emoji. "
    "Keep each reply under 300 characters. In each step either call exactly one tool, "
    "or write the next text message to the customer, never both."
)


class ToolSpec(BaseModel):
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class LLMRequest(BaseModel):
    agent_id: str
    system: str
    transcript: list[str]
    tools: list[ToolSpec]

    @property
    def key(self) -> str:
        """Stable hash of everything the model sees. Replay matches on it."""
        blob = json.dumps(
            {
                "system": self.system,
                "transcript": self.transcript,
                "tools": [t.name for t in self.tools],
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    @property
    def tail(self) -> str:
        return self.transcript[-1][:200] if self.transcript else ""


class ToolCallReply(BaseModel):
    name: str
    arguments: dict[str, Any]
    call_id: str


class LLMReply(BaseModel):
    text: str | None = None
    tool_call: ToolCallReply | None = None


class LLMCall(BaseModel):
    """One completion, as shown on the turn card and stored in fixtures."""

    key: str
    agent_id: str
    source: Literal["live", "replay", "replay-fallback", "tape"]
    model: str
    request_tail: str
    tools: list[str]
    reply: LLMReply
    duration_ms: int
    api_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    attempts: int = 1
    recorded_ms: int | None = None


def build_request(chat_ctx: llm.ChatContext, tools: list[llm.Tool]) -> LLMRequest:
    messages, extra = chat_ctx.to_provider_format(format="anthropic")
    system = "\n\n".join(extra.system_messages or [])
    match = AGENT_TAG.search(system)
    schemas = llm.ToolContext(tools).parse_function_tools("anthropic") if tools else []
    return LLMRequest(
        agent_id=match.group(1) if match else "unknown",
        system=system,
        transcript=transcript_lines(messages),
        tools=[
            ToolSpec(
                name=s["name"],
                description=s.get("description", ""),
                input_schema=s.get("input_schema", {}),
            )
            for s in schemas
        ],
    )


def transcript_lines(messages: list[dict[str, Any]]) -> list[str]:
    """Flatten Anthropic-format messages into readable, stable lines."""
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")
        if isinstance(content, str):
            lines.append(f"[{role}] {content}")
            continue
        for block in content or []:
            kind = block.get("type")
            if kind == "text":
                lines.append(f"[{role}] {block.get('text', '')}")
            elif kind == "tool_use":
                args = json.dumps(block.get("input"), sort_keys=True)
                lines.append(f"[{role} called {block.get('name')}] {args}")
            elif kind == "tool_result":
                out = block.get("content")
                if isinstance(out, list):
                    out = " ".join(x.get("text", "") for x in out if isinstance(x, dict))
                lines.append(f"[tool result] {out}")
    return lines


class CompletionLLM(llm.LLM[Any]):
    """Base class: subclasses implement ``complete``; this class handles LiveKit."""

    def __init__(self, model: str) -> None:
        super().__init__()
        self._model_name = model
        self.calls: list[LLMCall] = []

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "durable-text"

    @abstractmethod
    async def complete(self, req: LLMRequest) -> LLMCall: ...

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = False,
        tool_choice: NotGivenOr[llm.ToolChoice] = None,  # type: ignore[assignment]
        extra_kwargs: NotGivenOr[dict[str, Any]] = None,  # type: ignore[assignment]
    ) -> llm.LLMStream:
        # No LiveKit-level retries: providers retry with their own backoff.
        opts = APIConnectOptions(max_retry=0, retry_interval=0, timeout=conn_options.timeout)
        return _CompletionStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=opts)


class _CompletionStream(llm.LLMStream):
    _llm: CompletionLLM

    async def _run(self) -> None:
        req = build_request(self._chat_ctx, list(self._tools))
        started = time.monotonic()
        try:
            call = await self._llm.complete(req)
        except APIConnectionError:
            raise
        except Exception as exc:  # surfaced to LiveKit as a non-retryable error
            raise APIConnectionError(f"LLM call failed: {exc}", retryable=False) from exc
        call.duration_ms = call.duration_ms or int((time.monotonic() - started) * 1000)
        self._llm.calls.append(call)
        chunk_id = str(uuid.uuid4())
        if call.reply.tool_call is not None:
            tc = call.reply.tool_call
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id=chunk_id,
                    delta=llm.ChoiceDelta(
                        role="assistant",
                        tool_calls=[
                            llm.FunctionToolCall(
                                name=tc.name,
                                arguments=json.dumps(tc.arguments),
                                call_id=tc.call_id,
                            )
                        ],
                    ),
                )
            )
        elif call.reply.text:
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id=chunk_id,
                    delta=llm.ChoiceDelta(role="assistant", content=call.reply.text),
                )
            )
        if call.input_tokens is not None or call.output_tokens is not None:
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id=chunk_id,
                    usage=llm.CompletionUsage(
                        completion_tokens=call.output_tokens or 0,
                        prompt_tokens=call.input_tokens or 0,
                        total_tokens=(call.input_tokens or 0) + (call.output_tokens or 0),
                    ),
                )
            )
