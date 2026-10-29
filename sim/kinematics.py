from dataclasses import dataclass

from rewind.contracts import TimeSpan


@dataclass(frozen=True, slots=True)
class Belt:
    speed: float
    # Sorted and non-overlapping; the belt stands still during each one.
    jams: tuple[TimeSpan, ...] = ()

    def distance(self, t: float) -> float:
        stopped = sum(max(0.0, min(t, jam.end) - jam.start) for jam in self.jams if jam.start < t)
        return self.speed * (t - stopped)

    def time_at(self, distance: float) -> float:
        # Earliest time the belt has moved this far; a negative distance lies before the footage.
        if distance <= 0:
            return distance / self.speed
        t, moved = 0.0, 0.0
        for jam in self.jams:
            reach = moved + self.speed * (jam.start - t)
            if distance <= reach:
                return t + (distance - moved) / self.speed
            t, moved = jam.end, reach
        return t + (distance - moved) / self.speed

    def keys(self, duration_s: float) -> list[tuple[float, float]]:
        # The belt moves linearly between these (time, distance) points, so they are its keyframes.
        times = {0.0, duration_s}
        for jam in self.jams:
            times.update(t for t in (jam.start, jam.end) if 0 <= t <= duration_s)
        return [(t, self.distance(t)) for t in sorted(times)]


def merge_spans(spans: list[TimeSpan]) -> tuple[TimeSpan, ...]:
    merged: list[TimeSpan] = []
    for span in sorted(spans, key=lambda s: s.start):
        if merged and span.start <= merged[-1].end:
            last = merged.pop()
            span = TimeSpan(start=last.start, end=max(last.end, span.end))
        merged.append(span)
    return tuple(merged)
