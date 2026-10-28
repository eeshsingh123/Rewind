import hashlib
import hmac
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rewind.contracts import Attachment, CanonicalMessage, ChannelCapabilities, Reply
from rewind.core.conversation import AttachmentTooLargeError
from rewind.core.ports import BlobStore, ChannelError

logger = logging.getLogger(__name__)

GRAPH_BASE = "https://graph.facebook.com/v24.0"
# Cloud API limits: videos up to 16 MB, 3 reply buttons with 20-character titles, 1024
# characters of body text on a message with buttons.
MEDIA_LIMIT_BYTES = 16 * 1024 * 1024
CAPABILITIES = ChannelCapabilities(
    max_text_chars=4096, max_media_bytes=MEDIA_LIMIT_BYTES, max_buttons=3
)
_BUTTON_TITLE_CHARS = 20
_BUTTON_BODY_CHARS = 1024
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


class _Message(_Model):
    id: str
    sender: str = Field(alias="from")
    timestamp: str
    type: str
    text: _Text | None = None
    video: _Media | None = None
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

    @property
    def name(self) -> str:
        return "whatsapp"

    @property
    def capabilities(self) -> ChannelCapabilities:
        return CAPABILITIES

    async def send(self, reply: Reply) -> None:
        # One video with buttons and no other text goes as one message: the video is its header.
        joined = len(reply.media) == 1 and bool(reply.buttons) and not reply.text
        header: dict[str, Any] | None = None
        for media in reply.media:
            media_id = await self._upload(await self._blobs.local_path(media.blob_key))
            if joined:
                header = {"type": "video", "video": {"id": media_id}}
                continue
            video: dict[str, Any] = {"id": media_id}
            if media.caption:
                video["caption"] = media.caption
            await self._message(reply.chat_id, {"type": "video", "video": video})
        if reply.buttons:
            buttons = [
                {"type": "reply", "reply": {"id": b.value, "title": b.label[:_BUTTON_TITLE_CHARS]}}
                for b in reply.buttons[: CAPABILITIES.max_buttons]
            ]
            body = reply.text or (reply.media[0].caption if joined else None) or "…"
            interactive: dict[str, Any] = {
                "type": "button",
                "body": {"text": body[:_BUTTON_BODY_CHARS]},
                "action": {"buttons": buttons},
            }
            if header is not None:
                interactive["header"] = header
            await self._message(reply.chat_id, {"type": "interactive", "interactive": interactive})
        elif reply.text:
            text = {"body": reply.text[: CAPABILITIES.max_text_chars]}
            await self._message(reply.chat_id, {"type": "text", "text": text})

    async def show_progress(self, chat_id: str, key: str, text: str) -> None:
        sent = self._progress.get((chat_id, key), 0)
        if sent >= _PROGRESS_MESSAGES:
            return
        self._progress[(chat_id, key)] = sent + 1
        await self._message(chat_id, {"type": "text", "text": {"body": text}})

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

    async def _upload(self, path: Path) -> str:
        with path.open("rb") as file:
            result = await self._post(
                f"{GRAPH_BASE}/{self._phone_number_id}/media",
                data={"messaging_product": "whatsapp", "type": "video/mp4"},
                files={"file": (path.name, file, "video/mp4")},
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
    if message.text is not None:
        text = message.text.body
    elif message.interactive is not None and message.interactive.button_reply is not None:
        text = message.interactive.button_reply.id
    elif message.video is not None:
        text = message.video.caption
        attachments = (_attachment(message.video, "video"),)
    elif message.document is not None:
        is_video = (message.document.mime_type or "").startswith("video/")
        text = message.document.caption
        attachments = (_attachment(message.document, "video" if is_video else "file"),)
    if not text and not attachments:
        return None
    return CanonicalMessage(
        channel="whatsapp", message_id=message.id, chat_id=message.sender,
        sender_id=message.sender, received_at=datetime.fromtimestamp(int(message.timestamp), UTC),
        text=text, attachments=attachments,
    )  # fmt: skip


def _attachment(media: _Media, kind: Literal["video", "file"]) -> Attachment:
    return Attachment(kind=kind, ref=media.id, mime_type=media.mime_type, filename=media.filename)


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

    # D-7: answer 200 at once; receive() only claims the id and queues a job.
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
