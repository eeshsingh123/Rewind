import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import httpx

from rewind.contracts import Charge, Observation
from rewind.core.ports import (
    Completion,
    CompletionRequest,
    Segmentation,
    SegmentRequest,
)


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


def charge(unit: str, quantity: int = 100) -> Charge:
    return Charge(
        provider="fake", model="fake-1", unit=unit, quantity=quantity, usd=Decimal("0.01")
    )


class FakeLanguageModel:
    def __init__(self, replies: list[str | Exception], model_version: str = "fake-1") -> None:
        self.replies = replies
        self.requests: list[CompletionRequest] = []
        self._model_version = model_version

    @property
    def model_version(self) -> str:
        return self._model_version

    async def complete(self, request: CompletionRequest) -> Completion:
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Completion(text=reply, tool_calls=(), charges=(charge("output_token"),))


class FakeSegmenter:
    def __init__(self, find: Callable[[SegmentRequest], tuple[Observation, ...]]) -> None:
        self.find = find
        self.requests: list[SegmentRequest] = []

    @property
    def model_version(self) -> str:
        return "fake-sam"

    @property
    def max_frames(self) -> int:
        return 15_000

    @property
    def max_prompt_objects(self) -> int:
        return 16

    @property
    def min_billed_frames(self) -> int:
        return 1

    def estimate(self, request: SegmentRequest) -> tuple[Charge, ...]:
        return (charge("image", 1),)

    async def segment(self, request: SegmentRequest) -> Segmentation:
        self.requests.append(request)
        return Segmentation(observations=self.find(request), charges=(charge("image", 1),))


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
