import math
import shutil
import subprocess
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

import cv2
import pytest

from rewind.adapters.storage.local_disk import LocalDiskBlobStore
from rewind.adapters.storage.parquet_observations import ParquetObservationStore
from rewind.adapters.video.ffmpeg import FfmpegVideoTools
from rewind.contracts import (
    Box,
    Charge,
    Entity,
    ExitsView,
    Observation,
    RleMask,
    RuleSpec,
    TimeSpan,
)
from rewind.core.perception import (
    STRATEGIES,
    ActivitySample,
    IdleSkip,
    Ingest,
    PerceptionRunner,
    PlanInput,
    Planner,
    ProxyMaker,
    UnknownStrategyError,
    active_spans,
)
from rewind.core.ports import Segmentation, SegmentRequest, VideoInfo

pytestmark = pytest.mark.anyio

JAR_RULE = RuleSpec(
    entity=Entity(key="jar", phrase="glass jar"),
    parts={"cap": "jar lid"},
    checkpoint=ExitsView(),
)


class FakeVideo:
    def __init__(self) -> None:
        self.proxies = 0

    async def probe(self, source: Path) -> VideoInfo:
        return VideoInfo(duration_s=10.0, fps=25.0, width=1280, height=720, frame_count=250)

    async def make_proxy(
        self,
        source: Path,
        destination: Path,
        spans: Sequence[TimeSpan],
        fps: float,
        max_height: int,
    ) -> tuple[float, ...]:
        self.proxies += 1
        destination.write_bytes(b"proxy")
        return tuple(
            k / fps
            for span in spans
            for k in range(math.ceil(span.start * fps), math.floor(span.end * fps) + 1)
        )

    async def cut_clip(self, source: Path, destination: Path, span: TimeSpan) -> None:
        raise NotImplementedError


# Movement for the first four seconds of ten, then a stopped line.
class FakeActivity:
    async def sample(
        self, source: Path, sample_fps: float, regions: Sequence[Box]
    ) -> tuple[ActivitySample, ...]:
        return tuple(ActivitySample(t=i / 5, score=0.05 if i < 20 else 0.0) for i in range(1, 50))


# Finds one object on every other frame; bills every frame sent, with the 50-frame floor.
class FakeSegmenter:
    model_version = "sam-3.1"
    max_prompt_objects = 16
    min_billed_frames = 50

    def __init__(self, max_frames: int = 15_000) -> None:
        self.max_frames = max_frames
        self.requests: list[SegmentRequest] = []

    def estimate(self, request: SegmentRequest) -> tuple[Charge, ...]:
        frames = max(self.min_billed_frames, len(request.frame_times))
        return (
            Charge(
                provider="meta",
                model="sam-3.1",
                unit="video_frame",
                quantity=frames,
                usd=frames * Decimal("0.0002"),
            ),
        )

    async def segment(self, request: SegmentRequest) -> Segmentation:
        self.requests.append(request)
        observations = tuple(
            Observation(
                footage_id=request.footage_id,
                t=t,
                concept=request.prompt.text,  # type: ignore[union-attr]
                box=Box(x0=0.1, y0=0.2, x1=0.3, y1=0.4),
                score=0.9,
                object_id=0,
                mask=RleMask(height=2, width=2, counts=(1, 2, 1)),
            )
            for t in request.frame_times[::2]
        )
        return Segmentation(observations=observations, charges=self.estimate(request))


@pytest.fixture
def blobs(tmp_path: Path) -> LocalDiskBlobStore:
    return LocalDiskBlobStore(tmp_path / "blobs", "http://localhost", b"test")


async def test_rerunning_a_plan_costs_nothing_and_returns_the_same_observations(
    blobs: LocalDiskBlobStore, tmp_path: Path
) -> None:
    video = FakeVideo()
    proxies = ProxyMaker(blobs, video, max_height=720)
    ingest = Ingest(blobs, video, FakeActivity(), proxies, proxy_fps=2.0)
    runner = PerceptionRunner(proxies, ParquetObservationStore(blobs))
    segmenter = FakeSegmenter()
    await blobs.write("footage/line.mp4", b"original")

    ingested = await ingest.ingest("f1", "line1", "footage/line.mp4")
    plan = Planner(STRATEGIES).plan(
        "dense_track", PlanInput(ingested.footage, ingested.active_spans, (JAR_RULE,), fps=2.0)
    )
    first = await runner.run(ingested.footage, plan, segmenter)
    second = await runner.run(ingested.footage, plan, segmenter)

    assert ingested.active_spans == (TimeSpan(start=0.0, end=4.8),)
    assert [r.prompt.text for r in segmenter.requests] == ["glass jar", "jar lid"]  # type: ignore[union-attr]
    assert (first.paid_requests, second.paid_requests, second.cached_requests) == (2, 0, 2)
    assert second.charges == ()
    assert sorted(second.observations, key=repr) == sorted(first.observations, key=repr)
    assert video.proxies == 1


async def test_a_new_concept_on_processed_footage_pays_only_for_itself(
    blobs: LocalDiskBlobStore,
) -> None:
    video = FakeVideo()
    proxies = ProxyMaker(blobs, video, max_height=720)
    ingest = Ingest(blobs, video, FakeActivity(), proxies, proxy_fps=2.0)
    runner = PerceptionRunner(proxies, ParquetObservationStore(blobs))
    segmenter = FakeSegmenter()
    await blobs.write("footage/line.mp4", b"original")
    ingested = await ingest.ingest("f1", "line1", "footage/line.mp4")
    planner = Planner(STRATEGIES)

    def plan(*rules: RuleSpec):
        return planner.plan(
            "dense_track", PlanInput(ingested.footage, ingested.active_spans, rules, 2.0)
        )

    await runner.run(ingested.footage, plan(JAR_RULE), segmenter)
    label_rule = JAR_RULE.model_copy(update={"parts": {"label": "paper label"}})
    result = await runner.run(ingested.footage, plan(JAR_RULE, label_rule), segmenter)

    assert [r.prompt.text for r in segmenter.requests[2:]] == ["paper label"]  # type: ignore[union-attr]
    assert (result.paid_requests, result.cached_requests) == (1, 2)


async def test_a_partly_cached_window_pays_only_for_the_frames_not_seen(
    blobs: LocalDiskBlobStore,
) -> None:
    video = FakeVideo()
    proxies = ProxyMaker(blobs, video, max_height=720)
    runner = PerceptionRunner(proxies, ParquetObservationStore(blobs))
    segmenter = FakeSegmenter()
    await blobs.write("footage/line.mp4", b"original")
    ingested = await Ingest(blobs, video, FakeActivity(), proxies, 2.0).ingest(
        "f1", "line1", "footage/line.mp4"
    )
    planner = Planner(STRATEGIES)

    def plan(*spans: TimeSpan):
        return planner.plan("dense_track", PlanInput(ingested.footage, spans, (JAR_RULE,), 2.0))

    await runner.run(ingested.footage, plan(TimeSpan(start=0, end=5)), segmenter)
    wider = plan(TimeSpan(start=0, end=8), TimeSpan(start=20, end=22))
    result = await runner.run(ingested.footage, wider, segmenter)

    new_times = [r.frame_times for r in segmenter.requests[2:]]
    assert new_times == [(5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 20.0, 20.5, 21.0, 21.5, 22.0)] * 2
    jar_times = {o.t for o in result.observations if o.concept == "glass jar"}
    assert {0.0, 4.0, 5.5, 7.5, 20.0, 22.0} <= jar_times


async def test_windows_longer_than_the_frame_limit_are_split_into_requests(
    blobs: LocalDiskBlobStore,
) -> None:
    video = FakeVideo()
    proxies = ProxyMaker(blobs, video, max_height=720)
    runner = PerceptionRunner(proxies, ParquetObservationStore(blobs))
    segmenter = FakeSegmenter(max_frames=60)
    await blobs.write("footage/line.mp4", b"original")
    ingested = await Ingest(blobs, video, FakeActivity(), proxies, 2.0).ingest(
        "f1", "line1", "footage/line.mp4"
    )
    long_spans = (TimeSpan(start=0, end=40), TimeSpan(start=50, end=90))
    plan = Planner(STRATEGIES).plan(
        "dense_track", PlanInput(ingested.footage, long_spans, (JAR_RULE,), fps=2.0)
    )

    await runner.run(ingested.footage, plan, segmenter)

    assert all(len(r.frame_times) <= 60 for r in segmenter.requests)
    covered = {t for r in segmenter.requests if r.prompt.text == "glass jar" for t in r.frame_times}  # type: ignore[union-attr]
    assert len(covered) == 162


def test_idle_skip_pads_active_samples_and_bridges_short_gaps() -> None:
    samples = [
        ActivitySample(t=t, score=s) for t, s in [(1, 0.1), (2, 0), (3, 0.1), (9, 0.1), (20, 0)]
    ]

    spans = active_spans(
        samples, duration_s=20, idle_skip=IdleSkip(threshold=0.01, pad_s=1, merge_gap_s=2)
    )

    assert spans == (TimeSpan(start=0, end=4), TimeSpan(start=8, end=10))


def test_an_unknown_strategy_is_refused_by_name() -> None:
    with pytest.raises(UnknownStrategyError, match="dense_track"):
        Planner(STRATEGIES).plan("checkpoint", None)  # type: ignore[arg-type]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
async def test_each_proxy_frame_is_the_source_frame_its_time_map_names(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    # Frame n of the source has luma 16 + 3n, so a decoded proxy frame says which frame it is.
    pattern = "nullsrc=size=64x64:rate=12:duration=6,geq=lum='16+3*N':cb=128:cr=128"
    command = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", pattern, "-crf", "0", str(source)]
    subprocess.run(command, check=True)
    video = FfmpegVideoTools()
    spans = (TimeSpan(start=0.5, end=1.9), TimeSpan(start=4.0, end=5.0))

    frame_times = await video.make_proxy(
        source, tmp_path / "proxy.mp4", spans, fps=5, max_height=720
    )

    capture = cv2.VideoCapture(str(tmp_path / "proxy.mp4"))
    shown: list[float] = []
    while (frame := capture.read())[0]:
        # Decoding maps video luma 16..235 to 0..255; undo that to read the frame number back.
        luma = float(cv2.cvtColor(frame[1], cv2.COLOR_BGR2GRAY).mean()) * 219 / 255 + 16
        shown.append(round((luma - 16) / 3) / 12)
    capture.release()
    # A 12 fps source puts no frame on the 5 fps grid, so each step takes the next frame after it.
    expected = [math.ceil(round(g / 5 * 12, 6)) / 12 for g in (*range(3, 10), *range(20, 26))]
    assert list(frame_times) == pytest.approx(expected, abs=1e-3)
    assert shown == pytest.approx(frame_times, abs=1e-3)
