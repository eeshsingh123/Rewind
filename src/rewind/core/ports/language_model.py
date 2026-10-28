from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import JsonValue

from rewind.contracts import Charge


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    text: str


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    messages: tuple[ChatMessage, ...]
    max_output_tokens: int
    # A non-recursive JSON Schema the reply must follow (constrained decoding).
    response_schema: Mapping[str, JsonValue] | None = None


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    charges: tuple[Charge, ...]


class LanguageModelError(Exception):
    pass


class LanguageModel(Protocol):
    @property
    def model_version(self) -> str: ...

    async def complete(self, request: CompletionRequest) -> Completion: ...
