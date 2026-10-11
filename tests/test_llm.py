"""Provider layer: request mapping, replay, recording, SDK isolation options, retries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from livekit.agents import Agent, AgentSession, RunContext, function_tool, llm

from durable_text.envsafe import child_env, scrub_process_env
from durable_text.llm import claude_sdk
from durable_text.llm.base import LLMCall, LLMReply, LLMRequest, ToolSpec, build_request, transcript_lines
from durable_text.llm.claude_sdk import ClaudeSDKLLM, RetryableLLMError, _is_retryable
from durable_text.llm.replay import RecordingLLM, ReplayLLM, merge_recording
from durable_text.llm.scripted import ScriptedLLM


def req(tail: str = "[user] hi", tools: tuple[str, ...] = ("list_slots",)) -> LLMRequest:
    return LLMRequest(
        agent_id="scheduler",
        system='<agent id="scheduler"/> x',
        transcript=[tail],
        tools=[ToolSpec(name=t) for t in tools],
    )


def recorded(r: LLMRequest, text: str = "hello", ms: int = 4000) -> LLMCall:
    return LLMCall(
        key=r.key,
        agent_id=r.agent_id,
        source="live",
        model="claude-sonnet-5-5",
        request_tail=r.tail,
        tools=[],
        reply=LLMReply(text=text),
        duration_ms=ms,
        cost_usd=0.01,
    )


def test_request_key_is_stable_and_sensitive() -> None:
    assert req().key == req().key
    assert req().key != req("[user] hello").key
    assert req().key != req(tools=("hold_slot",)).key


def test_build_request_reads_agent_id_and_tools() -> None:
    ctx = llm.ChatContext()
    ctx.add_message(role="system", content='<agent id="front_desk"/>\nBe brief.')
    ctx.add_message(role="user", content="my dishwasher leaks")

    @function_tool
    async def list_slots(day: str) -> str:
        """List slots.

        Args:
            day: the day
        """
        return ""

    r = build_request(ctx, [list_slots])
    assert r.agent_id == "front_desk"
    assert r.transcript[-1] == "[user] my dishwasher leaks"
    assert r.tools[0].name == "list_slots" and "day" in r.tools[0].input_schema["properties"]


def test_transcript_lines_flatten_tool_use() -> None:
    lines = transcript_lines(
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "name": "list_slots", "input": {"day": "tue"}, "id": "x"}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "content": [{"type": "text", "text": "[]"}]}],
            },
        ]
    )
    assert lines == ["[user] hi", '[assistant called list_slots] {"day": "tue"}', "[tool result] []"]


def _fixtures(path: Path, calls: list[LLMCall]) -> Path:
    path.write_text(json.dumps({"calls": {c.key: c.model_dump() for c in calls}}))
    return path


async def test_replay_hit_returns_the_recorded_answer(tmp_path: Path) -> None:
    r = req()
    replay = ReplayLLM(_fixtures(tmp_path / "f.json", [recorded(r)]), scale=0)
    call = await replay.complete(r)
    assert (call.source, call.reply.text, call.recorded_ms) == ("replay", "hello", 4000)
    assert replay.size == 1 and replay.model == "claude-sonnet-5-5"


async def test_replay_miss_is_labelled_fallback(tmp_path: Path) -> None:
    replay = ReplayLLM(tmp_path / "missing.json", scale=0)
    call = await replay.complete(req("[user] any update on my repair?", ("check_booking",)))
    assert call.source == "replay-fallback"
    assert call.reply.tool_call is not None and call.reply.tool_call.name == "check_booking"


def test_fallback_answers_free_text_from_the_booking() -> None:
    """After a booking, unrecorded free text gets the booking, not a fresh slot offer."""
    from durable_text.llm.scripted import decide

    booked = [
        "[user] tuesday afternoon works",
        '[assistant called hold_slot] {"slot_id": "tue-1400"}',
        '[tool result] {"held": "tue-1400"}',
        '[assistant called request_dispatch] {"notes": "x"}',
        '[tool result] {"delegation": "d1", "status": "sent to dispatch"}',
        "[assistant] Done, that time is held.",
        "[user] thanks so much, when is the tech coming?",
    ]
    tools = [ToolSpec(name=t) for t in ("list_slots", "hold_slot", "request_dispatch", "check_booking")]
    first = decide(LLMRequest(agent_id="scheduler", system="s", transcript=booked, tools=tools))
    assert first.tool_call is not None and first.tool_call.name == "check_booking"

    result = {
        "held_window": "Tue 14:00-16:00",
        "dispatch": "technician confirmed",
        "technician": "Priya Nair",
    }
    after = [*booked, "[assistant called check_booking] {}", f"[tool result] {json.dumps(result)}"]
    reply = decide(LLMRequest(agent_id="scheduler", system="s", transcript=after, tools=tools))
    assert reply.text is not None
    assert "Priya Nair" in reply.text and "Tue 14:00-16:00" in reply.text
    assert "Tomorrow we have" not in reply.text

    pending = {"held_window": "Tue 14:00-16:00", "dispatch": "dispatch is picking a technician"}
    after = [*booked, "[assistant called check_booking] {}", f"[tool result] {json.dumps(pending)}"]
    reply = decide(LLMRequest(agent_id="scheduler", system="s", transcript=after, tools=tools))
    assert reply.text is not None and "picking a technician" in reply.text and "Priya" not in reply.text


async def test_recording_appends_and_merges_first_answer(tmp_path: Path) -> None:
    jsonl = tmp_path / "calls.jsonl"

    class Fixed(ScriptedLLM):
        async def complete(self, r: LLMRequest) -> LLMCall:
            return recorded(r, text=f"answer {len(self.calls)}")

    rec = RecordingLLM(Fixed(), jsonl)
    await rec.complete(req())
    await rec.complete(req())
    await rec.complete(req("[user] other"))
    out = tmp_path / "replay.json"
    assert merge_recording(jsonl, out) == 2
    data = json.loads(out.read_text())["calls"]
    assert data[req().key]["reply"]["text"] == "answer 0"


def test_sdk_options_isolate_every_query(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLAUDE_CODE_SOMETHING", "1")
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("SECRET_TOKEN", "x")
    opts = ClaudeSDKLLM("claude-haiku-5-5")._options(req(tools=("list_slots", "hold_slot")), str(tmp_path))
    assert opts.tools == []
    assert opts.allowed_tools == ["mcp__lk__list_slots", "mcp__lk__hold_slot"]
    assert opts.setting_sources == []
    assert opts.strict_mcp_config is True
    assert opts.permission_mode == "dontAsk"
    assert opts.max_turns == 2
    assert opts.cwd == str(tmp_path)
    assert opts.model == "claude-haiku-5-5"
    assert "no-session-persistence" in opts.extra_args
    assert set(opts.env) <= {"PATH", "HOME", "USER", "LANG", "TMPDIR", "ANTHROPIC_API_KEY"}
    assert list(opts.mcp_servers) == ["lk"]
    assert opts.hooks is not None and "PreToolUse" in opts.hooks


def test_child_env_and_scrub() -> None:
    src = {
        "PATH": "/bin",
        "HOME": "/h",
        "CLAUDE_CODE_X": "1",
        "CLAUDECODE": "1",
        "OTHER": "y",
        "ANTHROPIC_API_KEY": "",
    }
    assert child_env(src) == {"PATH": "/bin", "HOME": "/h"}
    import os

    saved = dict(os.environ)
    try:
        os.environ["CLAUDE_CODE_TEST"] = "1"
        os.environ["SOME_APP_SECRET"] = "x"
        removed = scrub_process_env()
        assert {"CLAUDE_CODE_TEST", "SOME_APP_SECRET"} <= set(removed)
        assert set(os.environ) <= {"PATH", "HOME", "USER", "LANG", "TMPDIR", "ANTHROPIC_API_KEY"}
    finally:
        os.environ.clear()
        os.environ.update(saved)
        os.environ.pop("CLAUDE_CODE_TEST", None)
        os.environ.pop("SOME_APP_SECRET", None)


@pytest.mark.parametrize(
    ("text", "status", "expected"),
    [
        ("Overloaded", None, True),
        ("rate limit reached", None, True),
        ("x", 529, True),
        ("x", 503, True),
        ("x", 429, True),
        ("invalid model", 400, False),
        ("bad input", None, False),
    ],
)
def test_retryable_errors(text: str, status: int | None, expected: bool) -> None:
    assert _is_retryable(text, status) is expected


async def test_retries_with_backoff_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[int] = []
    model = ClaudeSDKLLM(max_attempts=4, base_delay_s=0)

    async def flaky(r: LLMRequest) -> LLMCall:
        attempts.append(1)
        if len(attempts) < 3:
            raise RetryableLLMError("overloaded")
        return recorded(r)

    monkeypatch.setattr(model, "_once", flaky)
    call = await model.complete(req())
    assert call.attempts == 3 and len(attempts) == 3


async def test_gives_up_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    model = ClaudeSDKLLM(max_attempts=2, base_delay_s=0)

    async def always(r: LLMRequest) -> LLMCall:
        raise RetryableLLMError("overloaded")

    monkeypatch.setattr(model, "_once", always)
    with pytest.raises(claude_sdk.ClaudeSDKError, match="after 2 attempts"):
        await model.complete(req())


async def test_scripted_llm_drives_a_real_livekit_tool_loop() -> None:
    """The stream maps one tool call per completion into LiveKit's FunctionToolCall."""

    class Desk(Agent):
        def __init__(self) -> None:
            super().__init__(instructions='<agent id="scheduler"/> book visits')

        @function_tool
        async def list_slots(self, context: RunContext[Any], day: str) -> str:
            """List slots.

            Args:
                day: the day
            """
            return json.dumps(
                {"day": "tue", "open": [{"slot_id": "tue-1400", "window": "Tue 13 Oct 14:00-16:00"}]}
            )

    async with AgentSession(llm=ScriptedLLM()) as session:
        await session.start(Desk())
        result = await session.run(user_input="do you have anything tomorrow?")
        result.expect.next_event().is_function_call(name="list_slots", arguments={"day": "tomorrow"})
        result.expect.next_event().is_function_call_output()
        result.expect.next_event().is_message(role="assistant")
        result.expect.no_more_events()
