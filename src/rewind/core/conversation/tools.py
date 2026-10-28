from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from rewind.contracts import Button, ReplyMedia
from rewind.core.conversation.store import ChatRef
from rewind.core.ports import JobPayload, Tool


@dataclass(frozen=True, slots=True)
class TurnContext:
    workspace_id: str
    chat: ChatRef
    now: datetime


@dataclass(frozen=True, slots=True)
class Job:
    kind: str
    payload: JobPayload


@dataclass(frozen=True, slots=True)
class ToolResult:
    # What the model reads back; facts only, the agent words the reply.
    text: str
    buttons: tuple[Button, ...] = ()
    media: tuple[ReplyMedia, ...] = ()
    # Started only once the turn's reply is sent, so "started" never arrives after the results.
    jobs: tuple[Job, ...] = ()


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NoArguments(ToolArguments):
    pass


@dataclass(frozen=True, slots=True)
class ToolSpec[A: ToolArguments]:
    name: str
    description: str
    arguments: type[A]
    run: Callable[[TurnContext, A], Awaitable[ToolResult]]

    def as_tool(self) -> Tool:
        return Tool(self.name, self.description, self.arguments.model_json_schema())

    async def call(self, context: TurnContext, raw_arguments: str) -> ToolResult:
        try:
            arguments = self.arguments.model_validate_json(raw_arguments or "{}")
        except ValidationError as error:
            return ToolResult(f"error: invalid arguments for {self.name}: {error}")
        return await self.run(context, arguments)


class AgentTool(Protocol):
    @property
    def name(self) -> str: ...

    def as_tool(self) -> Tool: ...

    async def call(self, context: TurnContext, raw_arguments: str) -> ToolResult: ...


class ToolError(Exception):
    """Raised by a tool for a problem the user can fix; the message goes back to the model."""


@dataclass(frozen=True, slots=True)
class Toolbox:
    # Tools by name. A new tool is a new entry; the agent loop never changes.
    specs: Mapping[str, AgentTool] = field(default_factory=dict[str, AgentTool])

    @classmethod
    def of(cls, specs: Iterable[AgentTool]) -> "Toolbox":
        return cls({spec.name: spec for spec in specs})

    def as_tools(self) -> tuple[Tool, ...]:
        return tuple(spec.as_tool() for spec in self.specs.values())

    async def call(self, context: TurnContext, name: str, raw_arguments: str) -> ToolResult:
        spec = self.specs.get(name)
        if spec is None:
            return ToolResult(f"error: there is no tool named {name!r}")
        try:
            return await spec.call(context, raw_arguments)
        except ToolError as error:
            return ToolResult(f"error: {error}")
