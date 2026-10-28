import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import median

from rewind.contracts import Observation, RuleSpec, TimeSpan, Track, Zone
from rewind.core.rules.zones import center, overlap

# Parts may be sampled at other frames than the entity, so a part is matched to the nearest
# entity box within this gap.
MAX_PART_GAP_S = 0.5
# A part belongs to an entity when at least this share of the part's box lies inside it.
MIN_PART_OVERLAP = 0.5


@dataclass(frozen=True, slots=True)
class Item:
    track: Track
    path: tuple[Observation, ...]
    # The part observations attributed to this item, by part key, in time order.
    parts: Mapping[str, tuple[Observation, ...]]
    part_phrases: Mapping[str, str]
    zones: Mapping[str, Zone]

    @property
    def start(self) -> float:
        return self.path[0].t

    @property
    def end(self) -> float:
        return self.path[-1].t

    def seen_in(self, span: TimeSpan) -> tuple[Observation, ...]:
        return tuple(o for o in self.path if span.start <= o.t <= span.end)

    def parts_in(self, key: str, span: TimeSpan) -> tuple[Observation, ...]:
        return tuple(o for o in self.parts.get(key, ()) if span.start <= o.t <= span.end)


def build_items(
    spec: RuleSpec,
    tracks: Sequence[Track],
    observations: Sequence[Observation],
    zones: Mapping[str, Zone],
) -> tuple[Item, ...]:
    entity_tracks = sorted(
        (track for track in tracks if track.concept == spec.entity.phrase and moves(track)),
        key=lambda track: track.track_id,
    )
    paths = [tuple(sorted(track.observations, key=lambda o: o.t)) for track in entity_tracks]
    attributed: list[dict[str, list[Observation]]] = [{} for _ in entity_tracks]
    for key, phrase in sorted(spec.parts.items()):
        for observation in sorted((o for o in observations if o.concept == phrase), key=_order):
            owner = _owner(observation, paths)
            if owner is not None:
                attributed[owner].setdefault(key, []).append(observation)
    return tuple(
        Item(
            track=track,
            path=path,
            parts={key: tuple(found) for key, found in parts.items()},
            part_phrases=spec.parts,
            zones=zones,
        )
        for track, path, parts in zip(entity_tracks, paths, attributed, strict=True)
    )


# An item on a line travels; something that never leaves its own footprint is background (a
# parcel on a shelf, a bin by the belt). Measured against the object's own size, not the frame,
# so it holds for any zoom and any line speed.
def moves(track: Track) -> bool:
    boxes = [o.box for o in track.observations]
    size = median(math.hypot(b.x1 - b.x0, b.y1 - b.y0) for b in boxes)
    start = center(boxes[0])
    reach = max(math.hypot(c.x - start.x, c.y - start.y) for c in map(center, boxes))
    return reach >= size / 2


def _order(observation: Observation) -> tuple[float, float, float]:
    return (observation.t, observation.box.x0, observation.box.y0)


def _owner(part: Observation, paths: Sequence[tuple[Observation, ...]]) -> int | None:
    best: int | None = None
    best_overlap = MIN_PART_OVERLAP
    for index, path in enumerate(paths):
        nearest = _nearest(path, part.t)
        if nearest is None:
            continue
        share = overlap(part.box, nearest.box)
        if share > best_overlap or (best is None and share == best_overlap):
            best, best_overlap = index, share
    return best


def _nearest(path: tuple[Observation, ...], t: float) -> Observation | None:
    index = bisect_left(path, t, key=lambda o: o.t)
    candidates = path[max(index - 1, 0) : index + 1]
    nearest = min(candidates, key=lambda o: abs(o.t - t))
    if abs(nearest.t - t) > MAX_PART_GAP_S:
        return None
    return nearest
