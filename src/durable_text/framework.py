"""A small durable layer over plain LiveKit ``Agent`` classes.

Developer API::

    class Scheduler(DurableAgent):
        agent_id = "scheduler"

        def instructions_for(self, state: TurnState) -> str: ...

        @function_tool
        async def request_dispatch(self, ctx: RunContext[TurnState], notes: str) -> str:
            job = delegate(ctx.userdata, to="dispatch", task={...})
            return f"Dispatch request {job} sent."

        @on_event("dispatch.approved")
        async def confirmed(self, state: TurnState, event: WakeEvent) -> str:
            schedule(state, "visit_reminder", at=start - timedelta(hours=2))
            return "Text the customer that the visit is confirmed."

State is data, not coroutines: a turn rebuilds the agent from a checkpoint
(agent id, ``ChatContext.to_dict()``, ``values``), runs, and returns its side
effects as ``Effects``. The runtime commits the new checkpoint and the effects in
one transaction, so a crash at any point leaves the last committed turn intact.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, TypeVar

from livekit.agents import Agent, llm
from pydantic import BaseModel, Field

from . import clock


class WakeEvent(BaseModel):
    """Something that wakes a thread with no inbound text (an approval, a timer)."""

    type: str
    data: dict[str, Any] = Field(default_factory=dict)


class DelegationSpec(BaseModel):
    id: str
    to: str
    task: dict[str, Any]


class TimerSpec(BaseModel):
    kind: str
    due: str  # ISO minutes on the simulated clock
    payload: dict[str, Any] = Field(default_factory=dict)


class Proposal(BaseModel):
    technician_id: str
    technician_name: str
    slot_id: str
    window: str
    reason: str


class Effects(BaseModel):
    """Side effects a turn wants. Nothing here happens unless the turn commits."""

    holds: list[str] = Field(default_factory=list)
    delegations: list[DelegationSpec] = Field(default_factory=list)
    timers: list[TimerSpec] = Field(default_factory=list)
    proposal: Proposal | None = None


@dataclass
class TurnState:
    """``AgentSession.userdata`` for one turn. ``values`` is persisted, ``effects`` is not."""

    thread_id: str
    conn: sqlite3.Connection
    now: datetime
    customer: dict[str, Any]
    values: dict[str, Any] = field(default_factory=dict)
    effects: Effects = field(default_factory=Effects)
    delegation_seq: int = 0


Handler = Callable[[Any, TurnState, WakeEvent], Awaitable[str | None]]
_REGISTRY: dict[str, type[DurableAgent]] = {}
F = TypeVar("F", bound=Callable[..., Any])


def on_event(event_type: str) -> Callable[[F], F]:
    """Mark a method as the handler for a wake event, e.g. ``timer.visit_reminder``.

    The handler runs at the start of a wake turn and returns instructions for the
    agent's first text, or ``None`` to stay silent.
    """

    def deco(fn: F) -> F:
        fn.__durable_event__ = event_type  # type: ignore[attr-defined]
        return fn

    return deco


def delegate(state: TurnState, *, to: str, task: dict[str, Any]) -> str:
    """Ask another (headless) agent to do a task in its own session.

    Returns the delegation id at once. The work starts only if this turn commits;
    its result comes back later as a wake event (after a human approves it).
    """
    if to not in _REGISTRY:
        raise ValueError(f"unknown agent {to!r}")
    state.delegation_seq += 1
    spec = DelegationSpec(id=f"d{state.delegation_seq}", to=to, task=task)
    state.effects.delegations.append(spec)
    return spec.id


def schedule(
    state: TurnState, kind: str, *, at: datetime, payload: dict[str, Any] | None = None
) -> TimerSpec:
    """Durable timer on the simulated clock. Fires a ``timer.<kind>`` wake event."""
    spec = TimerSpec(kind=kind, due=clock.fmt_iso(at), payload=payload or {})
    state.effects.timers.append(spec)
    return spec


class DurableAgent(Agent):
    agent_id: ClassVar[str]
    headless: ClassVar[bool] = False
    _handlers: ClassVar[dict[str, str]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        handlers: dict[str, str] = {}
        for klass in reversed(cls.__mro__):
            for name, value in vars(klass).items():
                event = getattr(value, "__durable_event__", None)
                if event:
                    handlers[event] = name
        cls._handlers = handlers
        if "agent_id" in vars(cls):
            _REGISTRY[cls.agent_id] = cls

    def __init__(
        self,
        state: TurnState,
        *,
        chat_ctx: llm.ChatContext | None = None,
        resumed: bool = False,
        wake: WakeEvent | None = None,
    ) -> None:
        super().__init__(
            id=self.agent_id,
            instructions=f'<agent id="{self.agent_id}"/>\n{self.instructions_for(state)}',
            chat_ctx=chat_ctx,
        )
        self._resumed = resumed
        self._wake = wake

    def instructions_for(self, state: TurnState) -> str:
        raise NotImplementedError

    def handles(self, event_type: str) -> bool:
        return event_type in self._handlers

    async def on_enter(self) -> None:
        if self._wake is not None:
            event, self._wake = self._wake, None
            name = self._handlers.get(event.type)
            if name is None:
                return
            instructions = await getattr(self, name)(self.session.userdata, event)
            if instructions:
                self.session.generate_reply(instructions=instructions)
            return
        if self._resumed:
            # Rebuilt from a checkpoint: the user's text drives this turn, not a greeting.
            return
        # Handed to mid-turn: pick up the conversation right away.
        self.session.generate_reply()

    def handoff_to(self, target: type[DurableAgent], state: TurnState) -> DurableAgent:
        """Return the next agent with this agent's history (LiveKit does the switch).

        The handoff call itself has no output yet when this runs, so it is left
        out of the copy instead of reaching the next agent as a dangling call.
        """
        ctx = self.chat_ctx.copy()
        answered = {i.call_id for i in ctx.items if i.type == "function_call_output"}
        ctx.items[:] = [
            i for i in ctx.items if i.type != "function_call" or i.call_id in answered
        ]
        return target(state, chat_ctx=ctx)


def agent_class(agent_id: str) -> type[DurableAgent]:
    try:
        return _REGISTRY[agent_id]
    except KeyError:
        raise KeyError(f"no DurableAgent registered as {agent_id!r}") from None


def registered() -> dict[str, type[DurableAgent]]:
    return dict(_REGISTRY)
