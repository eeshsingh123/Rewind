import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import httpx

from rewind.contracts import Box, Charge, Observation, TimeSpan
from rewind.core.ask import Mark, Tile
from rewind.core.ask.names import NAMES_SCHEMA
from rewind.core.ask.reading import READ_SCHEMA
from rewind.core.ask.tools import CONFIRM, PLACE
from rewind.core.perception import ActivitySample
from rewind.core.ports import (
    Completion,
    CompletionRequest,
    Segmentation,
    SegmentRequest,
    ToolCall,
    VideoInfo,
    VideoToolsError,
)

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


def completion(text: str = "", *calls: ToolCall, output_tokens: int = 50) -> Completion:
    spent = Charge(
        provider="meta", model="muse", unit="output_token", quantity=output_tokens,
        usd=Decimal(0),
    )  # fmt: skip
    return Completion(text=text, tool_calls=calls, charges=(spent,))


PLAIN_READING = {
    "thing": "", "conditions": [], "combine": "none", "part": "", "plain": "", "unclear": "",
    "choices": [],
}  # fmt: skip


# Readings are scripted apart from the replies, since every question gets one.
@dataclass
class ScriptedModel:
    replies: list[Completion]
    requests: list[CompletionRequest] = field(default_factory=list[CompletionRequest])
    model_version: str = "muse-test"
    readings: list[dict[str, object]] = field(default_factory=list[dict[str, object]])

    # What Muse suggests calling a thing, which tiles it says are not that thing, and which are
    # not in the place asked about.
    names: dict[str, list[str]] = field(default_factory=dict[str, list[str]])
    not_real: list[str] = field(default_factory=list[str])
    not_there: list[str] = field(default_factory=list[str])

    async def complete(self, request: CompletionRequest) -> Completion:
        if request.response_schema is READ_SCHEMA:
            return completion(json.dumps(self.readings.pop(0) if self.readings else PLAIN_READING))
        last = request.messages[-1]
        if request.response_schema is NAMES_SCHEMA:
            asked = next((n for t, n in self.names.items() if t in getattr(last, "text", "")), [])
            return completion(json.dumps({"names": asked}))
        if CONFIRM in getattr(last, "text", ""):
            no = [{"tile": t, "verdict": "no", "answer": "not one"} for t in self.not_real]
            return completion(json.dumps({"tiles": no}))
        if PLACE in getattr(last, "text", ""):
            no = [{"tile": t, "verdict": "no", "answer": "beside it"} for t in self.not_there]
            return completion(json.dumps({"tiles": no}))
        self.requests.append(request)
        return self.replies.pop(0) if self.replies else completion("ok")


class FakeVideo:
    def __init__(self) -> None:
        self.drawn: list[tuple[Mark, ...]] = []
        self.unreadable = False

    async def probe(self, source: Path) -> VideoInfo:
        if self.unreadable:
            raise VideoToolsError("moov atom not found")
        return VideoInfo(duration_s=10.0, fps=25.0, width=1280, height=720, frame_count=250)

    async def make_proxy(
        self, source: Path, destination: Path, spans: Sequence[TimeSpan], fps: float,
        max_height: int,
    ) -> tuple[float, ...]:  # fmt: skip
        destination.write_bytes(b"proxy")
        return tuple(
            k / fps
            for span in spans
            for k in range(math.ceil(span.start * fps), math.floor(span.end * fps) + 1)
        )

    async def cut_clip(self, source: Path, destination: Path, span: TimeSpan) -> None:
        destination.write_bytes(b"clip")

    async def playable(self, source: Path, destination: Path) -> None:
        destination.write_bytes(b"playable")

    async def still(self, source: Path, t: float, destination: Path) -> None:
        destination.write_bytes(b"still")

    async def make_clip(
        self, source: Path, destination: Path, span: TimeSpan, marks: Sequence[Mark]
    ) -> None:
        self.drawn.append(tuple(marks))
        destination.write_bytes(b"clip with boxes")


class FakeActivity:
    async def sample(self, source: Path, sample_fps: float) -> tuple[ActivitySample, ...]:
        return tuple(ActivitySample(t=i / 5, score=0.05 if i < 30 else 0.0) for i in range(1, 50))


# Two cartons ride left and leave the view at its left edge at t=3; only the second carries a
# label. Video tracks the cartons; the label is looked for on stills.
class FakeSam:
    model_version = "sam-test"
    max_frames = 1000
    max_prompt_objects = 16
    max_tracked_objects = None
    min_billed_frames = 1

    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def estimate(self, request: SegmentRequest) -> tuple[Charge, ...]:
        return (billed_frames(len(request.frame_times)),)

    async def segment(self, request: SegmentRequest) -> Segmentation:
        self.calls += 1
        self.prompts.append(getattr(request.prompt, "text", ""))
        concept = self.prompts[-1]
        found: list[Observation] = []
        for t in request.frame_times:
            if t > 3:
                continue
            x = max(0.0, 0.6 - 0.2 * t)
            if concept == "carton":
                found += [
                    seen(request, concept, t, 1, x, 0.1),
                    seen(request, concept, t, 2, x, 0.6),
                ]
            if concept == "shipping label":
                found.append(seen(request, concept, t, 1, x + 0.1, 0.65, size=0.1))
        return Segmentation(tuple(found), (billed_frames(len(request.frame_times)),))


def billed_frames(count: int) -> Charge:
    return Charge(
        provider="meta", model="sam-test", unit="video_frame", quantity=count,
        usd=Decimal("0.0002") * count,
    )  # fmt: skip


def seen(
    request: SegmentRequest, concept: str, t: float, object_id: int, x: float, y: float,
    size: float = 0.3,
) -> Observation:  # fmt: skip
    box = Box(x0=x, y0=y, x1=x + size, y1=y + size)
    return Observation(
        footage_id=request.footage_id, t=t, concept=concept, box=box, score=0.9,
        object_id=object_id,
    )  # fmt: skip


class FakeSheet:
    def __init__(self) -> None:
        self.tiles: list[tuple[Tile, ...]] = []

    async def make_sheet(self, source: Path, destination: Path, tiles: Sequence[Tile]) -> None:
        self.tiles.append(tuple(tiles))
        destination.write_bytes(b"\xff\xd8 jpeg")
