from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import JsonValue

from rewind.contracts import Charge

Effort = Literal["minimal", "low", "medium", "high"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    # Raw JSON text as the model sent it; providers do not guarantee it fits the schema.
    arguments: str


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    # JSON Schema of the arguments.
    parameters: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class SystemMessage:
    text: str
    role: Literal["system"] = "system"


@dataclass(frozen=True, slots=True)
class UserMessage:
    text: str
    # Blob keys of images; the adapter resolves them.
    image_keys: tuple[str, ...] = ()
    role: Literal["user"] = "user"


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    role: Literal["assistant"] = "assistant"


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    tool_call_id: str
    text: str
    role: Literal["tool"] = "tool"


ChatMessage = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    messages: tuple[ChatMessage, ...]
    max_output_tokens: int
    # A non-recursive JSON Schema the reply must follow (constrained decoding).
    response_schema: Mapping[str, JsonValue] | None = None
    tools: tuple[Tool, ...] = ()
    # Reasoning tokens are most of the cost of a small call, so the default is the cheapest.
    effort: Effort = "minimal"


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    tool_calls: tuple[ToolCall, ...]
    # One charge per unit, each token counted once: input, cached_input, output, reasoning.
    charges: tuple[Charge, ...]


class LanguageModelError(Exception):
    pass


class LanguageModel(Protocol):
    @property
    def model_version(self) -> str: ...

    async def complete(self, request: CompletionRequest) -> Completion: ...
