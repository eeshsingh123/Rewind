from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rewind.contracts import TimeSpan


@dataclass(frozen=True, slots=True)
class ActivitySample:
    t: float
    # Share of the frame that changed, 0..1.
    score: float


class ActivityProbe(Protocol):
    async def sample(self, source: Path, sample_fps: float) -> tuple[ActivitySample, ...]: ...


@dataclass(frozen=True, slots=True)
class IdleSkip:
    sample_fps: float = 5.0
    threshold: float = 0.005
    # Kept around each active sample, so an item entering or leaving is not cut off.
    pad_s: float = 1.0
    # Idle gaps shorter than this are kept: a 50-frame floor makes tiny cuts worthless.
    merge_gap_s: float = 2.0


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
