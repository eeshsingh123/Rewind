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
    Observation,
    RleMask,
    SamplingPlan,
    SamplingWindow,
    TimeSpan,
)
from rewind.core.perception import ActivitySample, Ingest, PerceptionRunner, ProxyMaker
from rewind.core.ports import Segmentation, SegmentRequest, VideoInfo

pytestmark = pytest.mark.anyio


def _plan(footage_id: str, spans: Sequence[TimeSpan], *concepts: str) -> SamplingPlan:
    windows = tuple(SamplingWindow(span=s, fps=2.0, concepts=concepts) for s in spans)
    return SamplingPlan(footage_id=footage_id, windows=windows)


class FakeVideo:
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
        destination.write_bytes(b"proxy")
        return tuple(
            k / fps
            for span in spans
            for k in range(math.ceil(span.start * fps), math.floor(span.end * fps) + 1)
        )

    async def cut_clip(self, source: Path, destination: Path, span: TimeSpan) -> None:
        raise NotImplementedError

    async def playable(self, source: Path, destination: Path) -> None:
        raise NotImplementedError


# Movement for the first four seconds of ten, then a stopped line.
class FakeActivity:
    async def sample(self, source: Path, sample_fps: float) -> tuple[ActivitySample, ...]:
        return tuple(ActivitySample(t=i / 5, score=0.05 if i < 20 else 0.0) for i in range(1, 50))


# Finds one object on every other frame; bills every frame sent, with the 50-frame floor.
class FakeSegmenter:
    model_version = "sam-3.1"
    max_prompt_objects = 16
    max_tracked_objects: int | None = None
    min_billed_frames = 50

    max_frames = 15_000

    def __init__(self) -> None:
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
    jar_things = ("glass jar", "jar lid")
    spans = ingested.active_spans

    await runner.run(ingested.footage, _plan("f1", spans, *jar_things), segmenter)
    every = _plan("f1", spans, *jar_things, "paper label")
    result = await runner.run(ingested.footage, every, segmenter)

    assert [r.prompt.text for r in segmenter.requests[2:]] == ["paper label"]  # type: ignore[union-attr]
    assert (result.paid_requests, result.cached_requests) == (1, 2)


# Like SAM video: ids go to objects in order of appearance, and once `limit` objects were found,
# the request finds no new ones. Objects stand still at their own x and live from start to end.
class FullingSegmenter(FakeSegmenter):
    max_tracked_objects = 3
    lifetimes = ((0.0, 1.5), (1.0, 10.0), (2.0, 4.0), (4.5, 10.0), (6.0, 10.0))

    async def segment(self, request: SegmentRequest) -> Segmentation:
        self.requests.append(request)
        ids: dict[int, int] = {}
        found: list[Observation] = []
        for t in request.frame_times:
            for index, (start, end) in enumerate(self.lifetimes):
                if not start <= t <= end:
                    continue
                if index not in ids and len(ids) < 3:
                    ids[index] = 10 + len(ids)
                if index in ids:
                    found.append(
                        Observation(
                            footage_id=request.footage_id, t=t,
                            concept=request.prompt.text,  # type: ignore[union-attr]
                            box=Box(x0=index / 5, y0=0.4, x1=index / 5 + 0.1, y1=0.6),
                            score=0.9, object_id=ids[index],
                        )
                    )  # fmt: skip
        return Segmentation(observations=tuple(found), charges=self.estimate(request))


async def test_a_request_that_runs_out_of_slots_is_continued_with_the_same_ids(
    blobs: LocalDiskBlobStore,
) -> None:
    video = FakeVideo()
    proxies = ProxyMaker(blobs, video, max_height=720)
    runner = PerceptionRunner(proxies, ParquetObservationStore(blobs))
    segmenter = FullingSegmenter()
    await blobs.write("footage/line.mp4", b"original")
    ingested = await Ingest(blobs, video, FakeActivity(), proxies, 2.0).ingest(
        "f1", "line1", "footage/line.mp4"
    )
    plan = _plan("f1", (TimeSpan(start=0, end=10),), "glass jar", "jar lid")

    result = await runner.run(ingested.footage, plan, segmenter)

    jars = [o for o in result.observations if o.concept == "glass jar"]
    assert len({(o.t, o.box.x0) for o in jars}) == len(jars), "a shared frame is kept once"
    ids_by_object = {
        round(x * 5): {o.object_id for o in jars if o.box.x0 == x} for x in {o.box.x0 for o in jars}
    }
    assert ids_by_object == {0: {0}, 1: {1}, 2: {2}, 3: {3}, 4: {4}}
    starts = [r.frame_times[0] for r in segmenter.requests if r.prompt.text == "glass jar"]  # type: ignore[union-attr]
    assert sorted(starts) == [0.0, 2.0, 4.5, 5.0, 6.0]
    assert max(o.t for o in jars if o.box.x0 == 0.8) == 10.0


# Like SAM video on a long clip: it notices only what is in view as a request starts.
class StartOnlySegmenter(FakeSegmenter):
    arrives = (0.0, 7.0)

    async def segment(self, request: SegmentRequest) -> Segmentation:
        self.requests.append(request)
        first = request.frame_times[0]
        found = [
            Observation(
                footage_id=request.footage_id, t=t, concept="box",
                box=Box(x0=index / 5, y0=0.4, x1=index / 5 + 0.1, y1=0.6),
                score=0.9, object_id=index,
            )
            for t in request.frame_times
            for index, arrives in enumerate(self.arrives)
            if arrives <= first + 3 and t >= arrives
        ]  # fmt: skip
        return Segmentation(observations=tuple(found), charges=self.estimate(request))


async def test_what_arrives_during_a_long_clip_is_found(blobs: LocalDiskBlobStore) -> None:
    video = FakeVideo()
    proxies = ProxyMaker(blobs, video, max_height=720)
    runner = PerceptionRunner(proxies, ParquetObservationStore(blobs))
    await blobs.write("footage/line.mp4", b"original")
    ingested = await Ingest(blobs, video, FakeActivity(), proxies, 2.0).ingest(
        "f1", "line1", "footage/line.mp4"
    )
    plan = _plan("f1", (TimeSpan(start=0, end=10),), "box")

    result = await runner.run(ingested.footage, plan, StartOnlySegmenter())

    assert {o.object_id for o in result.observations if o.box.x0 == 0.2} == {1}
    assert {o.object_id for o in result.observations if o.box.x0 == 0.0} == {0}


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
