import hashlib
import hmac
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rewind.contracts import Attachment, Button, CanonicalMessage, ChannelCapabilities, Reply
from rewind.core.ask import attachment_kind
from rewind.core.ports import ActivityKind, AttachmentTooLargeError, BlobStore, ChannelError

logger = logging.getLogger(__name__)

GRAPH_BASE = "https://graph.facebook.com/v26.0"
# Cloud API limits: videos up to 16 MB and images up to 5 MB, captions of 1024 characters. Reply
# buttons fit no question (3 of them, 20 characters), so choices go as one list message: 10 rows
# of a 24-character title and a 72-character description.
MEDIA_LIMIT_BYTES = 16 * 1024 * 1024
CAPABILITIES = ChannelCapabilities(
    max_text_chars=4096,
    max_media_bytes=MEDIA_LIMIT_BYTES,
    max_buttons=10,
    max_image_bytes=5 * 1024 * 1024,
    can_edit=False,
)
_CAPTION_CHARS = 1024
_ROW_TITLE_CHARS = 24
_ROW_DESCRIPTION_CHARS = 72
_LIST_BUTTON = "Pick a question"
_LIST_SECTION = "Questions"
_MEDIA_TYPES = {"video": "video/mp4", "image": "image/jpeg"}
# The typing indicator lasts up to 25 s; it is not sent again sooner than this.
_TYPING_EVERY_S = 20.0
# WhatsApp cannot edit a sent message, so a job's progress is at most this many messages.
_PROGRESS_MESSAGES = 2
WEBHOOK_PATH = "/whatsapp/webhook"


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _Text(_Model):
    body: str


class _Media(_Model):
    id: str
    mime_type: str | None = None
    caption: str | None = None
    filename: str | None = None


class _ButtonReply(_Model):
    id: str
    title: str


class _Interactive(_Model):
    type: str
    button_reply: _ButtonReply | None = None
    list_reply: _ButtonReply | None = None


class _Message(_Model):
    id: str
    sender: str = Field(alias="from")
    timestamp: str
    type: str
    text: _Text | None = None
    video: _Media | None = None
    image: _Media | None = None
    audio: _Media | None = None
    document: _Media | None = None
    interactive: _Interactive | None = None


class _Value(_Model):
    messages: tuple[_Message, ...] = ()


class _Change(_Model):
    value: _Value


class _Entry(_Model):
    changes: tuple[_Change, ...] = ()


class _Webhook(_Model):
    entry: tuple[_Entry, ...] = ()


class _MediaInfo(_Model):
    url: str
    file_size: int | None = None


class WhatsAppChannel:
    def __init__(
        self, client: httpx.AsyncClient, token: str, phone_number_id: str, blobs: BlobStore
    ) -> None:
        self._client = client
        self._headers = {"Authorization": f"Bearer {token}"}
        self._phone_number_id = phone_number_id
        self._blobs = blobs
        self._progress: dict[tuple[str, str], int] = {}
        # Per chat: the message the indicator was last shown for, and when.
        self._typing: dict[str, tuple[str, float]] = {}

    @property
    def name(self) -> str:
        return "whatsapp"

    @property
    def capabilities(self) -> ChannelCapabilities:
        return CAPABILITIES

    async def send(self, reply: Reply) -> None:
        text = reply.text
        for media in reply.media:
            path = await self._blobs.local_path(media.blob_key)
            shown: dict[str, Any] = {"id": await self._upload(path, _MEDIA_TYPES[media.kind])}
            caption = media.caption or ""
            if len(caption) > _CAPTION_CHARS:
                text = f"{caption}\n\n{text}".strip()
            elif caption:
                shown["caption"] = caption
            await self._message(reply.chat_id, {"type": media.kind, media.kind: shown})
        if reply.buttons:
            chosen = reply.buttons[: CAPABILITIES.max_buttons]
            await self._message(reply.chat_id, _list(text or "…", chosen))
        elif text:
            body = {"body": text[: CAPABILITIES.max_text_chars]}
            await self._message(reply.chat_id, {"type": "text", "text": body})

    async def show_progress(
        self, chat_id: str, key: str, text: str, buttons: tuple[Button, ...] = ()
    ) -> None:
        # Buttons wait for a tap, so a message that has them is always sent.
        if buttons:
            await self.send(Reply(channel=self.name, chat_id=chat_id, text=text, buttons=buttons))
            return
        sent = self._progress.get((chat_id, key), 0)
        if sent >= _PROGRESS_MESSAGES:
            return
        self._progress[(chat_id, key)] = sent + 1
        await self._message(chat_id, {"type": "text", "text": {"body": text}})

    # The Cloud API ties its typing indicator to marking one received message as read.
    async def show_activity(
        self, chat_id: str, kind: ActivityKind, message_id: str | None = None
    ) -> None:
        if message_id is None:
            return
        now = time.monotonic()
        shown = self._typing.get(chat_id)
        if shown is not None and shown[0] == message_id and now - shown[1] < _TYPING_EVERY_S:
            return
        self._typing[chat_id] = (message_id, now)
        body = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
            "typing_indicator": {"type": "text"},
        }
        await self._post(f"{GRAPH_BASE}/{self._phone_number_id}/messages", json=body)

    async def fetch_attachment(self, attachment: Attachment, destination: Path) -> None:
        if (attachment.size_bytes or 0) > MEDIA_LIMIT_BYTES:
            raise AttachmentTooLargeError(MEDIA_LIMIT_BYTES)
        info = _MediaInfo.model_validate(await self._get_json(f"{GRAPH_BASE}/{attachment.ref}"))
        if (info.file_size or 0) > MEDIA_LIMIT_BYTES:
            raise AttachmentTooLargeError(MEDIA_LIMIT_BYTES)
        try:
            async with self._client.stream("GET", info.url, headers=self._headers) as response:
                response.raise_for_status()
                with destination.open("wb") as file:
                    async for chunk in response.aiter_bytes():
                        file.write(chunk)
        except httpx.HTTPError as error:
            raise ChannelError(f"whatsapp media download failed: {type(error).__name__}") from error

    async def _message(self, chat_id: str, content: dict[str, Any]) -> None:
        body = {"messaging_product": "whatsapp", "to": chat_id, **content}
        await self._post(f"{GRAPH_BASE}/{self._phone_number_id}/messages", json=body)

    async def _upload(self, path: Path, mime_type: str) -> str:
        with path.open("rb") as file:
            result = await self._post(
                f"{GRAPH_BASE}/{self._phone_number_id}/media",
                data={"messaging_product": "whatsapp", "type": mime_type},
                files={"file": (path.name, file, mime_type)},
            )
        return str(result["id"])

    async def _get_json(self, url: str) -> Any:
        try:
            response = await self._client.get(url, headers=self._headers, timeout=30)
        except httpx.HTTPError as error:
            raise ChannelError(f"whatsapp request failed: {type(error).__name__}") from error
        return _result(response)

    async def _post(self, url: str, **kwargs: Any) -> Any:
        try:
            response = await self._client.post(url, headers=self._headers, timeout=120, **kwargs)
        except httpx.HTTPError as error:
            raise ChannelError(f"whatsapp request failed: {type(error).__name__}") from error
        return _result(response)


def _list(body: str, buttons: tuple[Button, ...]) -> dict[str, Any]:
    interactive = {
        "type": "list",
        "body": {"text": body[: CAPABILITIES.max_text_chars]},
        "action": {
            "button": _LIST_BUTTON,
            "sections": [{"title": _LIST_SECTION, "rows": [_row(b) for b in buttons]}],
        },
    }
    return {"type": "interactive", "interactive": interactive}


# A question rarely fits a row's title, so it runs on into the description.
def _row(button: Button) -> dict[str, str]:
    label = button.label
    row = {"id": button.value, "title": label[:_ROW_TITLE_CHARS]}
    if len(label) <= _ROW_TITLE_CHARS:
        return row
    cut = label.rfind(" ", 1, _ROW_TITLE_CHARS + 1)
    cut = cut if cut > 0 else _ROW_TITLE_CHARS
    return {
        "id": button.value,
        "title": label[:cut],
        "description": label[cut:].strip()[:_ROW_DESCRIPTION_CHARS],
    }


def _result(response: httpx.Response) -> Any:
    try:
        payload = response.json()
    except ValueError as error:
        raise ChannelError(f"whatsapp: HTTP {response.status_code}") from error
    if response.is_error:
        message = payload.get("error", {}).get("message", "unknown error")
        raise ChannelError(f"whatsapp: HTTP {response.status_code}: {message}")
    return payload


def signature_valid(app_secret: str, body: bytes, header: str | None) -> bool:
    if not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header.removeprefix("sha256="))


def parse_webhook(raw: dict[str, Any]) -> tuple[CanonicalMessage, ...]:
    webhook = _Webhook.model_validate(raw)
    messages = (
        _canonical(message)
        for entry in webhook.entry
        for change in entry.changes
        for message in change.value.messages
    )
    return tuple(message for message in messages if message is not None)


def _canonical(message: _Message) -> CanonicalMessage | None:
    text: str | None = None
    attachments: tuple[Attachment, ...] = ()
    media = message.video or message.image or message.audio or message.document
    if message.text is not None:
        text = message.text.body
    elif message.interactive is not None:
        reply = message.interactive.list_reply or message.interactive.button_reply
        text = reply.id if reply is not None else None
    elif media is not None:
        text = media.caption
        kind = attachment_kind(media.mime_type, media.filename)
        attachments = (
            Attachment(kind=kind, ref=media.id, mime_type=media.mime_type, filename=media.filename),
        )
    if not text and not attachments:
        return None
    return CanonicalMessage(
        channel="whatsapp", message_id=message.id, chat_id=message.sender,
        sender_id=message.sender, received_at=datetime.fromtimestamp(int(message.timestamp), UTC),
        text=text, attachments=attachments,
    )  # fmt: skip


def webhook_router(
    verify_token: str, app_secret: str, receive: Callable[[CanonicalMessage], Awaitable[bool]]
) -> APIRouter:
    router = APIRouter()

    @router.get(WEBHOOK_PATH)
    async def verify(
        mode: str = Query(alias="hub.mode"),
        token: str = Query(alias="hub.verify_token"),
        challenge: str = Query(alias="hub.challenge"),
    ) -> Response:
        if mode != "subscribe" or not hmac.compare_digest(token, verify_token):
            raise HTTPException(status_code=403)
        return Response(challenge, media_type="text/plain")

    # Answer 200 at once; receive() only claims the id and queues a job.
    @router.post(WEBHOOK_PATH)
    async def deliver(request: Request) -> Response:
        body = await request.body()
        if not signature_valid(app_secret, body, request.headers.get("x-hub-signature-256")):
            raise HTTPException(status_code=401)
        try:
            messages = parse_webhook(await request.json())
        except (ValidationError, ValueError):
            logger.warning("skipped a whatsapp webhook of an unknown shape")
            return Response(status_code=200)
        for message in messages:
            await receive(message)
        return Response(status_code=200)

    return router


# The quick-tunnel URL changes on every start, so the app points the webhook at it again.
async def register_webhook(
    client: httpx.AsyncClient, app_id: str, app_secret: str, callback_url: str, verify_token: str
) -> None:
    try:
        response = await client.post(
            f"{GRAPH_BASE}/{app_id}/subscriptions",
            data={
                "object": "whatsapp_business_account",
                "callback_url": callback_url,
                "verify_token": verify_token,
                "fields": "messages",
                "access_token": f"{app_id}|{app_secret}",
            },
            timeout=60,
        )
    except httpx.HTTPError as error:
        message = f"whatsapp webhook registration failed: {type(error).__name__}"
        raise ChannelError(message) from error
    _result(response)


# Delivers this WhatsApp account's messages to the app; without it the webhook stays silent.
async def subscribe_account(client: httpx.AsyncClient, token: str, account_id: str) -> None:
    try:
        response = await client.post(
            f"{GRAPH_BASE}/{account_id}/subscribed_apps",
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        )
    except httpx.HTTPError as error:
        message = f"whatsapp account subscription failed: {type(error).__name__}"
        raise ChannelError(message) from error
    _result(response)
