from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from pydantic import JsonValue

from rewind.core.ports import ChatRef


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    role: Literal["user", "assistant"]
    text: str


class ConversationStore(Protocol):
    async def workspace_for_chat(self, chat: ChatRef) -> str | None: ...

    async def bind_chat(self, chat: ChatRef, workspace_id: str) -> None:
        """Upserts, so binding a chat again moves it to the new workspace."""
        ...

    async def chats(self, workspace_id: str) -> tuple[ChatRef, ...]: ...

    async def put_token(
        self, kind: str, token: str, payload: dict[str, JsonValue], expires_at: datetime
    ) -> None: ...

    async def take_token(self, kind: str, token: str, now: datetime) -> dict[str, JsonValue] | None:
        """The payload once, if the token exists and has not expired; later calls get None."""
        ...

    async def peek_token(
        self, kind: str, token: str, now: datetime
    ) -> dict[str, JsonValue] | None: ...

    async def append_history(self, chat: ChatRef, entry: HistoryEntry, at: datetime) -> None: ...

    async def history(self, chat: ChatRef, limit: int) -> tuple[HistoryEntry, ...]:
        """The latest entries, oldest first."""
        ...
