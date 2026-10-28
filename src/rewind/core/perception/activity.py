from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rewind.contracts import Box, Polygon, TimeSpan, Tripwire, Zone

WHOLE_FRAME = Box(x0=0, y0=0, x1=1, y1=1)
# A tripwire has no area, so motion is measured in a band this wide around it.
TRIPWIRE_MARGIN = 0.05


@dataclass(frozen=True, slots=True)
class ActivitySample:
    t: float
    # Share of the watched area that changed, 0..1; the highest over all regions.
    score: float


class ActivityProbe(Protocol):
    async def sample(
        self, source: Path, sample_fps: float, regions: Sequence[Box]
    ) -> tuple[ActivitySample, ...]: ...


@dataclass(frozen=True, slots=True)
class IdleSkip:
    sample_fps: float = 5.0
    threshold: float = 0.005
    # Kept around each active sample, so an item entering or leaving is not cut off.
    pad_s: float = 1.0
    # Idle gaps shorter than this are kept: a 50-frame floor makes tiny cuts worthless.
    merge_gap_s: float = 2.0


def watched_regions(zones: Sequence[Zone]) -> tuple[Box, ...]:
    regions = tuple(box for zone in zones if (box := _zone_box(zone)) is not None)
    return regions or (WHOLE_FRAME,)


def active_spans(
    samples: Sequence[ActivitySample], duration_s: float, idle_skip: IdleSkip
) -> tuple[TimeSpan, ...]:
    spans: list[TimeSpan] = []
    for sample in samples:
        if sample.score < idle_skip.threshold:
            continue
        start = max(0.0, sample.t - idle_skip.pad_s)
        end = min(duration_s, sample.t + idle_skip.pad_s)
        if spans and start - spans[-1].end <= idle_skip.merge_gap_s:
            spans[-1] = TimeSpan(start=spans[-1].start, end=max(spans[-1].end, end))
        else:
            spans.append(TimeSpan(start=start, end=end))
    return tuple(spans)


def _zone_box(zone: Zone) -> Box | None:
    shape = zone.shape
    if isinstance(shape, Polygon):
        xs = [p.x for p in shape.points]
        ys = [p.y for p in shape.points]
        return Box(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))
    if isinstance(shape, Tripwire):
        xs = (shape.start.x, shape.end.x)
        ys = (shape.start.y, shape.end.y)
        return Box(
            x0=max(0.0, min(xs) - TRIPWIRE_MARGIN),
            y0=max(0.0, min(ys) - TRIPWIRE_MARGIN),
            x1=min(1.0, max(xs) + TRIPWIRE_MARGIN),
            y1=min(1.0, max(ys) + TRIPWIRE_MARGIN),
        )
    return None
