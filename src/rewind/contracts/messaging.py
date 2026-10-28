from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from rewind.contracts.base import Contract, Id, Key


class Attachment(Contract):
    kind: Literal["video", "image", "file"]
    # Provider-specific handle; only the channel that received it can fetch it.
    ref: Id
    mime_type: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    filename: str | None = None


class CanonicalMessage(Contract):
    channel: Key
    message_id: Id
    chat_id: Id
    sender_id: Id
    received_at: AwareDatetime
    text: str | None = None
    attachments: tuple[Attachment, ...] = ()

    @model_validator(mode="after")
    def _has_content(self) -> Self:
        if not self.text and not self.attachments:
            raise ValueError("a message needs text or at least one attachment")
        return self


class ReplyMedia(Contract):
    kind: Literal["video", "image"]
    blob_key: Id
    caption: str | None = None


class Button(Contract):
    label: str = Field(min_length=1)
    value: str = Field(min_length=1)


class Reply(Contract):
    channel: Key
    chat_id: Id
    text: str
    media: tuple[ReplyMedia, ...] = ()
    buttons: tuple[Button, ...] = ()


class ChannelCapabilities(Contract):
    max_text_chars: int = Field(gt=0)
    # 0 means the channel cannot send media, or cannot show buttons.
    max_media_bytes: int = Field(ge=0)
    max_buttons: int = Field(ge=0)
