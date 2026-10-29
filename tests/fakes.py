import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import NamedTuple

import httpx

from rewind.contracts import Attachment, Button, ChannelCapabilities, Charge, Reply
from rewind.core.conversation import ActivityKind, AttachmentTooLargeError
from rewind.core.ports import Completion, CompletionRequest, ToolCall


class FakeBlobStore:
    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    async def write(self, key: str, data: bytes) -> None:
        self.blobs[key] = data

    async def write_file(self, key: str, source: Path) -> None:
        self.blobs[key] = source.read_bytes()

    async def read(self, key: str) -> bytes:
        return self.blobs[key]

    async def local_path(self, key: str) -> Path:
        raise NotImplementedError

    async def exists(self, key: str) -> bool:
        return key in self.blobs

    def signed_url(self, key: str, expires_in_s: int) -> str:
        return f"https://blobs.test/{key}"


RECORDINGS = Path(__file__).parent / "recordings"


# Serves recorded real responses in order; with match, only to a byte-identical request body.
def replay(name: str, sent: list[httpx.Request], match: bool = True) -> httpx.AsyncClient:
    exchanges = json.loads((RECORDINGS / name).read_text(encoding="utf-8"))["exchanges"]

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        body = json.loads(request.content)
        for exchange in exchanges:
            if not match or exchange["request"]["json"] == body:
                exchanges.remove(exchange)
                headers = exchange["response_headers"]
                return httpx.Response(exchange["status"], json=exchange["body"], headers=headers)
        raise AssertionError(f"no recording for this request in {name}; re-record it")

    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


CAPABILITIES = ChannelCapabilities(max_text_chars=4096, max_media_bytes=50_000_000, max_buttons=8)


def completion(text: str = "", *calls: ToolCall, output_tokens: int = 50) -> Completion:
    spent = Charge(
        provider="meta", model="muse", unit="output_token", quantity=output_tokens,
        usd=Decimal(0),
    )  # fmt: skip
    return Completion(text=text, tool_calls=calls, charges=(spent,))


@dataclass
class ScriptedModel:
    replies: list[Completion]
    requests: list[CompletionRequest] = field(default_factory=list[CompletionRequest])
    model_version: str = "muse-test"

    async def complete(self, request: CompletionRequest) -> Completion:
        self.requests.append(request)
        return self.replies.pop(0) if self.replies else completion("ok")


class Shown(NamedTuple):
    # "send", "progress", "activity" or "fetch".
    kind: str
    chat_id: str
    text: str
    key: str = ""


@dataclass
class FakeChannel:
    name: str = "telegram"
    sent: list[Reply] = field(default_factory=list[Reply])
    # Everything the chat showed, in order.
    shown: list[Shown] = field(default_factory=list[Shown])
    fetch_limit: int = 20_000_000

    @property
    def capabilities(self) -> ChannelCapabilities:
        return CAPABILITIES

    async def send(self, reply: Reply) -> None:
        self.sent.append(reply)
        self.shown.append(Shown("send", reply.chat_id, reply.text))

    async def show_progress(
        self, chat_id: str, key: str, text: str, buttons: tuple[Button, ...] = ()
    ) -> None:
        self.sent.append(Reply(channel=self.name, chat_id=chat_id, text=text, buttons=buttons))
        self.shown.append(Shown("progress", chat_id, text, key))

    async def show_activity(self, chat_id: str, kind: ActivityKind) -> None:
        self.shown.append(Shown("activity", chat_id, kind))

    async def fetch_attachment(self, attachment: Attachment, destination: Path) -> None:
        if (attachment.size_bytes or 0) > self.fetch_limit:
            raise AttachmentTooLargeError(self.fetch_limit)
        self.shown.append(Shown("fetch", "", attachment.ref))
        destination.write_bytes(b"video")

    def progress(self, key: str) -> list[str]:
        return [s.text for s in self.shown if s.kind == "progress" and s.key == key]
