from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from rewind.contracts import Attachment, Button, ChannelCapabilities, Reply

type ActivityKind = Literal["typing", "upload_video"]


class ChannelError(Exception):
    pass


class AttachmentTooLargeError(ChannelError):
    """Raised by a channel when the provider will not hand over a file this big."""

    def __init__(self, limit_bytes: int) -> None:
        super().__init__(f"the channel can fetch files up to {limit_bytes} bytes")
        self.limit_bytes = limit_bytes


@dataclass(frozen=True, slots=True)
class ChatRef:
    channel: str
    chat_id: str


class Channel(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def capabilities(self) -> ChannelCapabilities: ...

    async def send(self, reply: Reply) -> None: ...

    async def fetch_attachment(self, attachment: Attachment, destination: Path) -> None: ...


class ChatChannel(Channel, Protocol):
    async def show_progress(
        self, chat_id: str, key: str, text: str, buttons: tuple[Button, ...] = ()
    ) -> None:
        """Updates the progress message with this key in place, or sends it if it is new."""
        ...

    async def show_activity(
        self, chat_id: str, kind: ActivityKind, message_id: str | None = None
    ) -> None:
        """Shows "typing…" or similar for a few seconds; message_id is what is being answered."""
        ...


class ChatStore(Protocol):
    async def workspace_for_chat(self, chat: ChatRef) -> str | None: ...

    async def bind_chat(self, chat: ChatRef, workspace_id: str) -> None:
        """Upserts, so binding a chat again moves it to the new workspace."""
        ...

    async def latest_session(self, chat: ChatRef) -> str | None: ...

    async def set_latest_session(self, chat: ChatRef, session_id: str) -> None: ...
