import asyncio
import math
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import JsonValue

from rewind.adapters.jobs.in_process import InProcessJobRunner
from rewind.adapters.storage.local_disk import LocalDiskBlobStore
from rewind.adapters.storage.parquet_observations import ParquetObservationStore
from rewind.adapters.storage.sql import SqlRepository
from rewind.contracts import (
    Attachment,
    Box,
    CanonicalMessage,
    Charge,
    Entity,
    ExitsView,
    Has,
    Line,
    Observation,
    Requirement,
    Rule,
    RuleSpec,
    TimeSpan,
)
from rewind.core.analysis import (
    ANALYZE_JOB,
    INGEST_JOB,
    Analysis,
    AnalysisSettings,
    Mark,
)
from rewind.core.budget import BudgetedSegmenter, BudgetLimits, BudgetScope, Ledger
from rewind.core.conversation import (
    INSTRUCTIONS,
    TURN_JOB,
    Agent,
    Analyze,
    AnalyzeSettings,
    ChatRef,
    Conversation,
    ConversationSettings,
    Inbox,
    Outbox,
    Readings,
    Toolbox,
    TurnLimits,
)
from rewind.core.perception import (
    STRATEGIES,
    ActivitySample,
    Ingest,
    PerceptionRunner,
    Planner,
    ProxyMaker,
)
from rewind.core.ports import (
    JobPayload,
    Segmentation,
    Segmenter,
    SegmentRequest,
    ToolCall,
    VideoInfo,
    VideoToolsError,
)
from rewind.core.rules import RuleCompiler
from tests.fakes import FakeChannel, ScriptedModel, completion

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 10, 29, 9, 0, tzinfo=UTC)
CARTON_RULE = RuleSpec(
    entity=Entity(key="carton", phrase="carton"),
    parts={"label": "shipping label"},
    checkpoint=ExitsView(),
    requirements=(Requirement(predicate=Has(part="label"), timing="at_checkpoint"),),
)


class FakeVideo:
    def __init__(self) -> None:
        self.drawn: list[tuple[Mark, ...]] = []
        # Holds reading a video back until set, to ask for a check meanwhile.
        self.gate: asyncio.Event | None = None
        self.unreadable = False

    async def probe(self, source: Path) -> VideoInfo:
        if self.gate is not None:
            await self.gate.wait()
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

    async def still(self, source: Path, t: float, destination: Path) -> None:
        destination.write_bytes(b"still")

    async def make_clip(
        self, source: Path, destination: Path, span: TimeSpan, marks: Sequence[Mark]
    ) -> None:
        self.drawn.append(tuple(marks))
        destination.write_bytes(b"clip with boxes")


class FakeActivity:
    async def sample(
        self, source: Path, sample_fps: float, regions: Sequence[Box]
    ) -> tuple[ActivitySample, ...]:
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
        return (_frames(len(request.frame_times)),)

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
                    _seen(request, concept, t, 1, x, 0.1),
                    _seen(request, concept, t, 2, x, 0.6),
                ]
            if concept == "shipping label":
                found.append(_seen(request, concept, t, 1, x + 0.1, 0.65, size=0.1))
        return Segmentation(tuple(found), (_frames(len(request.frame_times)),))


def _frames(count: int) -> Charge:
    return Charge(
        provider="meta", model="sam-test", unit="video_frame", quantity=count,
        usd=Decimal("0.0002") * count,
    )  # fmt: skip


def _seen(
    request: SegmentRequest, concept: str, t: float, object_id: int, x: float, y: float,
    size: float = 0.3,
) -> Observation:  # fmt: skip
    box = Box(x0=x, y0=y, x1=x + size, y1=y + size)
    return Observation(
        footage_id=request.footage_id, t=t, concept=concept, box=box, score=0.9,
        object_id=object_id,
    )  # fmt: skip


@dataclass
class Segmenters:
    sam: FakeSam
    ledger: Ledger

    def video(self, scope: BudgetScope) -> Segmenter:
        return BudgetedSegmenter(self.sam, self.ledger, scope)

    def image(self, scope: BudgetScope) -> Segmenter:
        return BudgetedSegmenter(self.sam, self.ledger, scope)


@dataclass
class Setup:
    repository: SqlRepository
    blobs: LocalDiskBlobStore
    sam: FakeSam
    channel: FakeChannel
    analysis: Analysis
    jobs: InProcessJobRunner
    line: Line
    rule: Rule
    video: FakeVideo


@pytest.fixture
async def setup(tmp_path: Path) -> AsyncIterator[Setup]:
    repository = SqlRepository.from_url(f"sqlite+aiosqlite:///{tmp_path / 'rewind.db'}")
    await repository.create_tables()
    blobs = LocalDiskBlobStore(tmp_path / "blobs", "https://tunnel.test", b"key")
    video = FakeVideo()
    proxies = ProxyMaker(blobs, video, 720)
    observations = ParquetObservationStore(blobs)
    sam = FakeSam()
    ledger = Ledger(repository, BudgetLimits(Decimal(5), Decimal(100)), clock=lambda: NOW)
    channel = FakeChannel()

    def no_compiler(workspace_id: str) -> RuleCompiler:
        raise AssertionError("no zone needs finding in these tests")

    analysis = Analysis(
        repository=repository, blobs=blobs,
        ingest=Ingest(blobs, video, FakeActivity(), proxies, 2.0),
        planner=Planner(STRATEGIES), runner=PerceptionRunner(proxies, observations),
        observations=observations, segmenters=Segmenters(sam, ledger),
        compiler_for=no_compiler, video=video, stills=video, clips=video,
        notifier=Outbox({"telegram": channel}),
        settings=AnalysisSettings(fps=2.0),
    )  # fmt: skip
    jobs = InProcessJobRunner()
    jobs.register(ANALYZE_JOB, analysis.handle_analyze)
    line = Line(id="line-1", workspace_id="ws", name="Packing line 2", created_at=NOW)
    await repository.save_line(line)
    rule = Rule(
        id="rule-1", line_id=line.id, text="Every carton needs a label before it leaves.",
        normalized_text="every carton needs a label before it leaves", spec=CARTON_RULE,
        created_at=NOW,
    )  # fmt: skip
    await repository.save_rule(rule)
    source = tmp_path / "belt.mp4"
    source.write_bytes(b"video bytes")
    await blobs.write_file("footage/f1/belt.mp4", source)
    await analysis.read(_payload(footage_id="f1", original_key="footage/f1/belt.mp4"))
    yield Setup(repository, blobs, sam, channel, analysis, jobs, line, rule, video)
    await repository.close()


def _payload(**extra: JsonValue) -> JobPayload:
    return {
        "workspace_id": "ws", "line_id": "line-1", "channel": "telegram", "chat_id": "chat",
        **extra,
    }  # fmt: skip


async def test_a_check_reports_each_rule_plainly_draws_the_proof_and_a_rerun_is_free(
    setup: Setup,
) -> None:
    analyze = _payload(footage_ids=["f1"], rule_ids=[setup.rule.id], job_id="job-1")

    await setup.analysis.handle_analyze(analyze)
    calls_after_first = setup.sam.calls
    await setup.analysis.handle_analyze({**analyze, "job_id": "job-2"})
    [violation] = await setup.repository.violations("f1")
    footage = await setup.repository.get_footage("f1")
    assert footage is not None
    clip = await setup.analysis.clip(footage, violation, 1, 1, 50_000_000)

    assert violation.reason == "no shipping label when it left the view"
    # Video follows the cartons; the label is looked for on one still, shared by both cartons.
    assert setup.sam.prompts == ["carton", "shipping label"]
    assert setup.sam.calls == calls_after_first == 2
    done = setup.channel.sent[-1]
    summary, buttons = done.text, done.buttons
    assert summary.startswith("Done in ")
    assert "for $0.00" in summary
    assert "Rule 1 · Every carton needs a label before it leaves." in summary
    assert "❌ 1 of 2 cartons that left the view failed: the carton at 00:03." in summary
    assert [b.label for b in buttons] == ["Show clips"]
    assert clip.caption == ("Rule 1 · clip 1 of 1\nThe carton at 00:03: no shipping label.")
    [marks] = setup.video.drawn
    assert sorted((m.kind, m.tag) for m in marks) == [
        ("flagged", "no shipping label"), ("passed", "")
    ]  # fmt: skip
    assert await setup.blobs.exists(clip.media[0].blob_key)


def _analyze(setup: Setup, readings: Readings, confirm_above_usd: Decimal) -> Analyze:
    return Analyze(
        setup.repository, setup.repository, setup.analysis, readings,
        Outbox({"telegram": setup.channel}), setup.jobs,
        AnalyzeSettings(confirm_above_usd=confirm_above_usd, daily_cap_usd=Decimal(5)),
        lambda: NOW,
    )  # fmt: skip


@dataclass
class Chat:
    inbox: Inbox
    model: ScriptedModel


# The whole chat path: turns and ingest run as jobs, as in the app, with the line's chat bound to
# the fixture's workspace.
async def _chat(setup: Setup, confirm_above_usd: Decimal) -> Chat:
    readings = Readings()
    analyze = _analyze(setup, readings, confirm_above_usd)
    model = ScriptedModel([])
    conversation = Conversation(
        repository=setup.repository, store=setup.repository, blobs=setup.blobs, jobs=setup.jobs,
        channels={"telegram": setup.channel},
        agent=Agent(INSTRUCTIONS, Toolbox.of([analyze.tool()]), TurnLimits()),
        model_for=lambda workspace_id: model, readings=readings,
        settings=ConversationSettings(frozenset(), "https://tunnel.test"), clock=lambda: NOW,
    )  # fmt: skip
    setup.jobs.register(TURN_JOB, conversation.handle_job)
    setup.jobs.register(INGEST_JOB, analyze.handle_ingest)
    await setup.repository.bind_chat(ChatRef("telegram", "chat"), "ws")
    return Chat(Inbox(setup.repository, setup.jobs), model)


async def _video_then_check_it(setup: Setup, chat: Chat) -> None:
    setup.video.gate = asyncio.Event()
    chat.model.replies = [
        completion("", ToolCall("c1", "analyze", "{}")),
        completion("Still reading the video; I'll check it as soon as it's ready."),
    ]
    video = Attachment(kind="video", ref="file-9", size_bytes=6_600_000, filename="clip.mp4")
    await chat.inbox.receive(_message(None, "m1", attachments=(video,)))
    await chat.inbox.receive(_message("Check it", "m2"))
    await _until(lambda: any(s.kind == "send" for s in setup.channel.shown))
    setup.video.gate.set()
    await setup.jobs.drain()


def _message(
    text: str | None, message_id: str, attachments: tuple[Attachment, ...] = ()
) -> CanonicalMessage:
    return CanonicalMessage(
        channel="telegram", message_id=message_id, chat_id="chat", sender_id="7",
        received_at=NOW, text=text, attachments=attachments,
    )  # fmt: skip


async def _until(condition: Callable[[], bool]) -> None:
    for _ in range(500):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timed out")


def _ack_key(channel: FakeChannel) -> str:
    return next(s.key for s in channel.shown if s.kind == "progress")


async def test_a_check_asked_for_while_the_video_is_read_starts_once_it_is_ready(
    setup: Setup,
) -> None:
    chat = await _chat(setup, confirm_above_usd=Decimal(1))

    await _video_then_check_it(setup, chat)

    state = chat.model.requests[0].messages[1].text
    tool_result = chat.model.requests[1].messages[-1].text
    assert "recording: clip.mp4 (being read)" in state
    assert tool_result.startswith("NOT started yet: clip.mp4 is still being read.")
    texts = [s.text for s in setup.channel.shown if s.kind in ("send", "progress")]
    ack, ready = setup.channel.progress(_ack_key(setup.channel))
    assert ack.startswith("Got the video for *Packing line 2*.")
    assert ready.startswith("Got it: 10 s of *Packing line 2* (clip.mp4, 1280x720).")
    assert "Checking it against 1 rule now (about $0." in ready
    # The agent's one reply comes between the ack and the ready message; the check comes last.
    assert texts[:3] == [
        ack,
        "Still reading the video; I'll check it as soon as it's ready.",
        ready,
    ]
    assert texts[3].startswith("Checking *clip.mp4*")
    assert texts[-1].startswith("Done in ")
    assert not any("no recordings" in text for text in texts)
