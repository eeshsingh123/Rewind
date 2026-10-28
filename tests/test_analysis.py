import math
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
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
    Box,
    Button,
    ChannelCapabilities,
    Charge,
    Entity,
    ExitsView,
    Has,
    Line,
    Observation,
    ReplyMedia,
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
    naive_tracks,
)
from rewind.core.budget import BudgetedSegmenter, BudgetLimits, BudgetScope, Ledger
from rewind.core.conversation import Analyze, AnalyzeSettings, ChatRef, TurnContext
from rewind.core.perception import (
    STRATEGIES,
    ActivitySample,
    Ingest,
    PerceptionRunner,
    Planner,
    ProxyMaker,
)
from rewind.core.ports import JobPayload, Segmentation, Segmenter, SegmentRequest, VideoInfo
from rewind.core.rules import RuleCompiler

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 10, 29, 9, 0, tzinfo=UTC)
CARTON_RULE = RuleSpec(
    entity=Entity(key="carton", phrase="carton"),
    parts={"label": "shipping label"},
    checkpoint=ExitsView(),
    requirements=(Requirement(predicate=Has(part="label"), timing="at_checkpoint"),),
)
CAPABILITIES = ChannelCapabilities(max_text_chars=4096, max_media_bytes=50_000_000, max_buttons=8)


class FakeVideo:
    async def probe(self, source: Path) -> VideoInfo:
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


class FakeActivity:
    async def sample(
        self, source: Path, sample_fps: float, regions: Sequence[Box]
    ) -> tuple[ActivitySample, ...]:
        return tuple(ActivitySample(t=i / 5, score=0.05 if i < 30 else 0.0) for i in range(1, 50))


# Two cartons leave the view by t=3; only the second carries a label.
class FakeSam:
    model_version = "sam-test"
    max_frames = 1000
    max_prompt_objects = 16
    min_billed_frames = 1

    def __init__(self) -> None:
        self.calls = 0

    def estimate(self, request: SegmentRequest) -> tuple[Charge, ...]:
        return (_frames(len(request.frame_times)),)

    async def segment(self, request: SegmentRequest) -> Segmentation:
        self.calls += 1
        concept = getattr(request.prompt, "text", "")
        found: list[Observation] = []
        for t in request.frame_times:
            if t > 3:
                continue
            if concept == "carton":
                found += [_seen(request, concept, t, 1, 0.1), _seen(request, concept, t, 2, 0.5)]
            if concept == "shipping label":
                found.append(_seen(request, concept, t, 1, 0.55, size=0.1))
        return Segmentation(tuple(found), (_frames(len(request.frame_times)),))


def _frames(count: int) -> Charge:
    return Charge(
        provider="meta", model="sam-test", unit="video_frame", quantity=count,
        usd=Decimal("0.0002") * count,
    )  # fmt: skip


def _seen(
    request: SegmentRequest, concept: str, t: float, object_id: int, x: float, size: float = 0.3
) -> Observation:
    box = Box(x0=x, y0=0.3, x1=x + size, y1=0.3 + size)
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
class Notes:
    sent: list[tuple[str, tuple[ReplyMedia, ...], tuple[Button, ...]]] = field(
        default_factory=list[tuple[str, tuple[ReplyMedia, ...], tuple[Button, ...]]]
    )

    async def notify(
        self, channel: str, chat_id: str, text: str, media: tuple[ReplyMedia, ...] = (),
        buttons: tuple[Button, ...] = (),
    ) -> None:  # fmt: skip
        self.sent.append((text, media, buttons))

    async def progress(self, channel: str, chat_id: str, key: str, text: str) -> None:
        pass


@dataclass
class Setup:
    repository: SqlRepository
    blobs: LocalDiskBlobStore
    sam: FakeSam
    notes: Notes
    analysis: Analysis
    jobs: InProcessJobRunner
    line: Line
    rule: Rule


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
    notes = Notes()

    def no_compiler(workspace_id: str) -> RuleCompiler:
        raise AssertionError("no zone needs finding in these tests")

    analysis = Analysis(
        repository=repository, blobs=blobs,
        ingest=Ingest(blobs, video, FakeActivity(), proxies, 2.0),
        planner=Planner(STRATEGIES), runner=PerceptionRunner(proxies, observations),
        observations=observations, segmenters=Segmenters(sam, ledger),
        compiler_for=no_compiler, video=video, stills=video, notifier=notes,
        settings=AnalysisSettings(fps=2.0),
    )  # fmt: skip
    jobs = InProcessJobRunner()
    jobs.register(INGEST_JOB, analysis.handle_ingest)
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
    await analysis.handle_ingest(_payload(footage_id="f1", original_key="footage/f1/belt.mp4"))
    yield Setup(repository, blobs, sam, notes, analysis, jobs, line, rule)
    await repository.close()


def _payload(**extra: JsonValue) -> JobPayload:
    return {
        "workspace_id": "ws", "line_id": "line-1", "channel": "telegram", "chat_id": "chat",
        **extra,
    }  # fmt: skip


def test_a_reused_object_id_after_a_gap_is_a_new_item() -> None:
    def seen(t: float) -> Observation:
        return Observation(
            footage_id="f", t=t, concept="carton", box=Box(x0=0, y0=0, x1=0.2, y1=0.2),
            score=0.9, object_id=7,
        )  # fmt: skip

    tracks = naive_tracks("f", [seen(0.0), seen(0.5), seen(5.0), seen(5.5)], max_gap_s=1.5)

    assert [(t.track_id, len(t.observations)) for t in tracks] == [
        ("carton#1", 2),
        ("carton#2", 2),
    ]


async def test_a_check_saves_violations_sends_a_clip_and_a_rerun_is_free(setup: Setup) -> None:
    analyze = _payload(
        footage_ids=["f1"], rule_ids=[setup.rule.id], job_id="job-1", max_media_bytes=50_000_000
    )

    await setup.analysis.handle_analyze(analyze)
    calls_after_first = setup.sam.calls
    await setup.analysis.handle_analyze({**analyze, "job_id": "job-2"})

    violations = await setup.repository.violations("f1")
    assert [(v.track_id, v.reason) for v in violations] == [
        ("carton#1", "no shipping label when it left the view")
    ]
    summary, _, _ = setup.notes.sent[-2]
    assert "carton 1: no shipping label when it left the view (rule 1)" in summary
    assert "for $0.00" in summary
    _, media, _ = setup.notes.sent[-1]
    assert await setup.blobs.exists(media[0].blob_key)
    assert setup.sam.calls == calls_after_first == 2


async def test_analyze_asks_for_a_tap_above_the_threshold_and_starts_once(setup: Setup) -> None:
    started: list[JobPayload] = []
    jobs = InProcessJobRunner()

    async def record(payload: JobPayload) -> None:
        started.append(payload)

    jobs.register(ANALYZE_JOB, record)
    analyze = Analyze(
        setup.repository, setup.repository, jobs, setup.analysis, {"telegram": CAPABILITIES},
        AnalyzeSettings(confirm_above_usd=Decimal("0.001"), daily_cap_usd=Decimal(5)),
    )  # fmt: skip
    context = TurnContext("ws", ChatRef("telegram", "chat"), NOW)

    result = await analyze.tool().call(context, "{}")
    token = result.buttons[0].value.split(" ")[1]
    await analyze.confirm(context, token)
    again = await analyze.confirm(context, token)
    await jobs.drain()

    assert "NOT started yet" in result.text
    assert len(started) == 1
    assert again.text.startswith("That button has expired")
