import asyncio
import html
import json
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rewind.contracts import Attachment, Button, CanonicalMessage, ChannelCapabilities, Reply
from rewind.core.ask import attachment_kind
from rewind.core.ports import ActivityKind, AttachmentTooLargeError, BlobStore, ChannelError

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"
# Bot API limits: getFile serves up to 20 MB; bots send videos up to 50 MB and photos up to 10 MB;
# callback data is 64 bytes.
FETCH_LIMIT_BYTES = 20 * 1024 * 1024
CAPABILITIES = ChannelCapabilities(
    max_text_chars=4096,
    max_media_bytes=50 * 1024 * 1024,
    max_buttons=8,
    max_image_bytes=10 * 1024 * 1024,
    can_edit=True,
)
# Telegram cuts a caption off at this many characters; a longer one goes as text after it.
CAPTION_LIMIT = 1024
POLL_TIMEOUT_S = 50
RETRY_AFTER_ERROR_S = 5
_BOLD = re.compile(r"\*([^*\n]+)\*")


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _User(_Model):
    id: int


class _Chat(_Model):
    id: int


class _File(_Model):
    file_id: str
    file_size: int | None = None
    mime_type: str | None = None
    file_name: str | None = None


class _Message(_Model):
    message_id: int
    date: int
    chat: _Chat
    sender: _User | None = Field(default=None, alias="from")
    text: str | None = None
    caption: str | None = None
    video: _File | None = None
    video_note: _File | None = None
    # Telegram's sizes of one photo, smallest first.
    photo: tuple[_File, ...] = ()
    voice: _File | None = None
    audio: _File | None = None
    document: _File | None = None


class _CallbackQuery(_Model):
    id: str
    sender: _User = Field(alias="from")
    message: _Message | None = None
    data: str | None = None


class _Update(_Model):
    update_id: int
    message: _Message | None = None
    callback_query: _CallbackQuery | None = None


class TelegramChannel:
    def __init__(self, client: httpx.AsyncClient, token: str, blobs: BlobStore) -> None:
        self._client = client
        # The Bot API puts the token in the path; the httpx logger must stay above INFO.
        self._base = f"{API_BASE}/bot{token}"
        self._file_base = f"{API_BASE}/file/bot{token}"
        self._blobs = blobs
        # Per (chat, key): the progress message's id and what it shows now.
        self._progress: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}

    @property
    def name(self) -> str:
        return "telegram"

    @property
    def capabilities(self) -> ChannelCapabilities:
        return CAPABILITIES

    async def send(self, reply: Reply) -> None:
        markup = _keyboard(reply.buttons[: CAPABILITIES.max_buttons])
        text = reply.text
        for index, media in enumerate(reply.media):
            path = await self._blobs.local_path(media.blob_key)
            caption = media.caption or ""
            if len(caption) > CAPTION_LIMIT:
                text = f"{caption}\n\n{text}".strip()
                caption = ""
            fields: dict[str, Any] = {"chat_id": reply.chat_id, "caption": _html(caption),
                                      "parse_mode": "HTML"}  # fmt: skip
            # With no text after it, the last proof carries the buttons: one message, not two.
            if markup is not None and not text and index == len(reply.media) - 1:
                fields["reply_markup"] = json.dumps(markup)
            if media.kind == "image":
                await self._call_with_file("sendPhoto", fields, "photo", path, "image/jpeg")
            else:
                await self._call_with_file("sendVideo", fields, "video", path, "video/mp4")
        if text:
            await self._call("sendMessage", _text_body(reply.chat_id, _html(text), markup))

    async def show_progress(
        self, chat_id: str, key: str, text: str, buttons: tuple[Button, ...] = ()
    ) -> None:
        markup = _keyboard(buttons[: CAPABILITIES.max_buttons])
        body = _text_body(chat_id, _html(text), markup)
        shown = self._progress.get((chat_id, key))
        if shown is not None:
            message_id, last = shown
            # Telegram refuses an edit that changes nothing.
            if body == last:
                return
            await self._call("editMessageText", body | {"message_id": message_id})
            self._progress[(chat_id, key)] = (message_id, body)
            return
        sent = await self._call("sendMessage", body)
        self._progress[(chat_id, key)] = (int(sent["message_id"]), body)

    # The "/" menu in the chat: (command, what it does), in order.
    async def set_commands(self, commands: Sequence[tuple[str, str]]) -> None:
        listed = [{"command": name, "description": text} for name, text in commands]
        await self._call("setMyCommands", {"commands": listed})

    async def show_activity(
        self, chat_id: str, kind: ActivityKind, message_id: str | None = None
    ) -> None:
        await self._call("sendChatAction", {"chat_id": chat_id, "action": kind})

    async def fetch_attachment(self, attachment: Attachment, destination: Path) -> None:
        if (attachment.size_bytes or 0) > FETCH_LIMIT_BYTES:
            raise AttachmentTooLargeError(FETCH_LIMIT_BYTES)
        try:
            found = await self._call("getFile", {"file_id": attachment.ref})
        except ChannelError as error:
            if "too big" in str(error):
                raise AttachmentTooLargeError(FETCH_LIMIT_BYTES) from error
            raise
        try:
            async with self._client.stream("GET", f"{self._file_base}/{found['file_path']}") as r:
                r.raise_for_status()
                with destination.open("wb") as file:
                    async for chunk in r.aiter_bytes():
                        file.write(chunk)
        except httpx.HTTPError as error:
            raise ChannelError(f"telegram file download failed: {type(error).__name__}") from error

    # Long polling: nothing needs a public URL. Unacknowledged updates come back after a
    # restart, which the receiver's dedupe drops.
    async def poll(self, receive: Callable[[CanonicalMessage], Awaitable[bool]]) -> None:
        offset = 0
        while True:
            try:
                updates = await self._call(
                    "getUpdates",
                    {"offset": offset, "timeout": POLL_TIMEOUT_S,
                     "allowed_updates": ["message", "callback_query"]},
                    timeout=POLL_TIMEOUT_S + 10,
                )  # fmt: skip
            except ChannelError:
                logger.exception("telegram getUpdates failed")
                await asyncio.sleep(RETRY_AFTER_ERROR_S)
                continue
            for raw in updates:
                offset = max(offset, int(raw["update_id"]) + 1)
                await self._dispatch(raw, receive)

    async def _dispatch(
        self, raw: dict[str, Any], receive: Callable[[CanonicalMessage], Awaitable[bool]]
    ) -> None:
        try:
            update = _Update.model_validate(raw)
        except ValidationError:
            logger.warning("skipped a telegram update of an unknown shape")
            return
        # Buttons stay after a tap: the other suggested questions can still be asked.
        if update.callback_query is not None:
            await self._call("answerCallbackQuery", {"callback_query_id": update.callback_query.id})
        message = _canonical(update)
        if message is not None:
            await receive(message)

    async def _call(self, method: str, body: dict[str, Any], timeout: float = 30) -> Any:
        try:
            response = await self._client.post(f"{self._base}/{method}", json=body, timeout=timeout)
        except httpx.HTTPError as error:
            raise ChannelError(f"telegram {method} failed: {type(error).__name__}") from error
        return _result(method, response)

    async def _call_with_file(
        self, method: str, fields: dict[str, Any], field: str, path: Path, mime_type: str
    ) -> Any:
        try:
            with path.open("rb") as file:
                response = await self._client.post(
                    f"{self._base}/{method}",
                    data={k: str(v) for k, v in fields.items()},
                    files={field: (path.name, file, mime_type)},
                    timeout=300,
                )
        except httpx.HTTPError as error:
            raise ChannelError(f"telegram {method} failed: {type(error).__name__}") from error
        return _result(method, response)


def parse_update(raw: dict[str, Any]) -> CanonicalMessage | None:
    return _canonical(_Update.model_validate(raw))


def _canonical(update: _Update) -> CanonicalMessage | None:
    query = update.callback_query
    if query is not None:
        if query.message is None or not query.data:
            return None
        return CanonicalMessage(
            channel="telegram", message_id=str(update.update_id),
            chat_id=str(query.message.chat.id), sender_id=str(query.sender.id),
            received_at=datetime.now(UTC), text=query.data,
        )  # fmt: skip
    message = update.message
    if message is None or message.sender is None:
        return None
    found = _found(message)
    attachments = () if found is None else (found,)
    text = message.text or message.caption
    if not text and not attachments:
        return None
    return CanonicalMessage(
        channel="telegram", message_id=str(update.update_id), chat_id=str(message.chat.id),
        sender_id=str(message.sender.id), received_at=datetime.fromtimestamp(message.date, UTC),
        text=text, attachments=attachments,
    )  # fmt: skip


def _found(message: _Message) -> Attachment | None:
    if message.photo:
        return _attachment(message.photo[-1], "image", "image/jpeg")
    if message.video is not None:
        return _attachment(message.video, "video", "video/mp4")
    if message.video_note is not None:
        return _attachment(message.video_note, "video", "video/mp4")
    sound = message.voice or message.audio
    if sound is not None:
        return _attachment(sound, "audio", "audio/ogg")
    if message.document is not None:
        file = message.document
        return _attachment(file, attachment_kind(file.mime_type, file.file_name), None)
    return None


def _attachment(
    file: _File, kind: Literal["video", "image", "audio", "file"], usual_type: str | None
) -> Attachment:
    return Attachment(
        kind=kind, ref=file.file_id, mime_type=file.mime_type or usual_type,
        size_bytes=file.file_size, filename=file.file_name,
    )  # fmt: skip


def _result(method: str, response: httpx.Response) -> Any:
    try:
        payload = response.json()
    except ValueError as error:
        raise ChannelError(f"telegram {method}: HTTP {response.status_code}") from error
    if not payload.get("ok"):
        raise ChannelError(f"telegram {method}: {payload.get('description', 'unknown error')}")
    return payload["result"]


# Replies mark bold with single asterisks; Telegram's HTML mode is the one that never breaks
# on a stray underscore in a file name.
def _html(text: str) -> str:
    escaped = html.escape(text[: CAPABILITIES.max_text_chars], quote=False)
    return _BOLD.sub(r"<b>\1</b>", escaped)


def _text_body(chat_id: str, text: str, markup: dict[str, Any] | None) -> dict[str, Any]:
    body: dict[str, Any] = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    if markup is not None:
        body["reply_markup"] = markup
    return body


def _keyboard(buttons: tuple[Button, ...]) -> dict[str, Any] | None:
    if not buttons:
        return None
    if any(len(b.value.encode()) > 64 for b in buttons):
        raise ChannelError("telegram button values are limited to 64 bytes")
    rows = [[{"text": b.label, "callback_data": b.value}] for b in buttons]
    return {"inline_keyboard": rows}
