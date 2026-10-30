import itertools
import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from rewind.contracts import TimeSpan

# A gap in output counts as a stop when it is this many times the usual gap between items, and
# at least MIN_STOP_S: measured against the footage's own pace, so it holds for any scene.
STOP_FACTOR = 3.0
MIN_STOP_S = 5.0
# Below this many items in a row there is no usual gap to compare with.
MIN_ITEMS_FOR_PACE = 4
# Items "stand still" in a stretch when, second by second, they move at under this share of the
# speed they usually move at (the upper quartile of their steps, which is the moving belt).
HELD_SHARE = 0.25
BIN_S = 1.0
MIN_STEPS = 8

# The (t, x, y) centres of one tracked item, in time order.
type Path = Sequence[tuple[float, float, float]]


class Stop(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    # From the last item out before the gap to the first one after it.
    gap: TimeSpan
    # The stretch seen standing still: nothing moved in view, or the items in view
    # stood still. A gap with neither may be items the camera check missed, so it is no stop.
    stood: TimeSpan
    seen: Literal["no_motion", "items_still"]
    usual_gap_s: float

    @property
    def lost_items(self) -> int:
        return round((self.stood.end - self.stood.start) / self.usual_gap_s)


# times: when each item left, from which the footage's own pace is read.
def find_stops(
    times: Sequence[float],
    active_spans: Sequence[TimeSpan],
    duration_s: float,
    paths: Sequence[Path] = (),
) -> tuple[Stop, ...]:
    ordered = sorted({round(t, 3) for t in times})
    usual = usual_gap(ordered)
    if usual is None:
        return ()
    limit = max(MIN_STOP_S, STOP_FACTOR * usual)
    stops: list[Stop] = []
    for start, end in itertools.pairwise(ordered):
        if end - start <= limit:
            continue
        gap = TimeSpan(start=start, end=end)
        still = _longest_still(gap, active_spans, duration_s)
        held = _held(gap, paths)
        if still is not None and still.end - still.start >= limit:
            stops.append(Stop(gap=gap, stood=still, seen="no_motion", usual_gap_s=usual))
        elif held is not None and held.end - held.start >= limit:
            stops.append(Stop(gap=gap, stood=held, seen="items_still", usual_gap_s=usual))
    return tuple(stops)


# The usual time between one item and the next; None with too few items to tell.
def usual_gap(times: Sequence[float]) -> float | None:
    ordered = sorted(set(times))
    if len(ordered) < MIN_ITEMS_FOR_PACE:
        return None
    usual = statistics.median(b - a for a, b in itertools.pairwise(ordered))
    return usual if usual > 0 else None


# The longest stretch inside the gap where the tracked items moved far slower than they usually
# do, measured against their own speed on this footage, so it holds for any scene and camera.
def _held(gap: TimeSpan, paths: Sequence[Path]) -> TimeSpan | None:
    steps = [(a[0], b[0], math.dist(a[1:], b[1:]) / (b[0] - a[0]))
             for path in paths for a, b in itertools.pairwise(path) if b[0] > a[0]]  # fmt: skip
    if len(steps) < MIN_STEPS:
        return None
    usual_dt = statistics.median(t2 - t1 for t1, t2, _ in steps)
    steps = [s for s in steps if s[1] - s[0] <= 2 * usual_dt]
    moving = statistics.quantiles([speed for _, _, speed in steps], n=4)[2]
    if moving <= 0:
        return None
    bins: defaultdict[int, list[float]] = defaultdict(list)
    for t1, t2, speed in steps:
        if gap.start <= t1 and t2 <= gap.end:
            bins[math.floor(t1 / BIN_S)].append(speed)
    slow = {b for b, speeds in bins.items() if statistics.median(speeds) < HELD_SHARE * moving}
    best: tuple[int, int] | None = None
    for first in sorted(slow):
        last = first
        while last + 1 in slow:
            last += 1
        if best is None or last - first > best[1] - best[0]:
            best = (first, last)
    if best is None:
        return None
    return TimeSpan(start=max(gap.start, best[0] * BIN_S), end=min(gap.end, (best[1] + 1) * BIN_S))


def _longest_still(
    gap: TimeSpan, active_spans: Sequence[TimeSpan], duration_s: float
) -> TimeSpan | None:
    idle: list[TimeSpan] = []
    cursor = 0.0
    for span in sorted(active_spans, key=lambda s: s.start):
        if span.start > cursor:
            idle.append(TimeSpan(start=cursor, end=span.start))
        cursor = max(cursor, span.end)
    if cursor < duration_s:
        idle.append(TimeSpan(start=cursor, end=duration_s))
    inside = [
        TimeSpan(start=max(gap.start, s.start), end=min(gap.end, s.end))
        for s in idle
        if min(gap.end, s.end) - max(gap.start, s.start) > 0
    ]
    if not inside:
        return None
    return max(inside, key=lambda s: s.end - s.start)
