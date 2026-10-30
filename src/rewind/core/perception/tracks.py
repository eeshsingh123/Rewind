from collections import defaultdict
from collections.abc import Sequence

from rewind.contracts import Box, Observation, Track

# Boxes of the same object on the same frame, seen by two requests, overlap at least this much.
SAME_OBJECT_IOU = 0.5
# Seen again later, an object's box is mostly inside or around its earlier box, even when
# something passing in front of it cut that box short.
_SAME_PLACE = 0.7


# Naive association for now: one track per segmenter object id and concept,
# split where the object is unseen for longer than max_gap_s, because SAM reuses ids.
def naive_tracks(
    footage_id: str, observations: Sequence[Observation], max_gap_s: float
) -> tuple[Track, ...]:
    by_object: defaultdict[tuple[str, int], list[Observation]] = defaultdict(list)
    for observation in observations:
        if observation.object_id is not None:
            by_object[(observation.concept, observation.object_id)].append(observation)
    runs: list[tuple[str, list[Observation]]] = []
    for (concept, _), seen in by_object.items():
        seen.sort(key=lambda o: o.t)
        run = [seen[0]]
        for observation in seen[1:]:
            if observation.t - run[-1].t > max_gap_s:
                runs.append((concept, run))
                run = []
            run.append(observation)
        runs.append((concept, run))
    runs.sort(key=lambda r: (r[0], r[1][0].t))
    numbers: defaultdict[str, int] = defaultdict(int)
    tracks: list[Track] = []
    for concept, run in runs:
        numbers[concept] += 1
        tracks.append(
            Track(
                footage_id=footage_id,
                track_id=f"{concept}#{numbers[concept]}",
                concept=concept,
                observations=tuple(run),
            )
        )
    return tuple(tracks)


# One track per item: tracks that are the same item, under two names or noticed twice, become one.
def distinct_tracks(tracks: Sequence[Track], rejoin_s: float) -> tuple[Track, ...]:
    kept: list[Track] = []
    for track in sorted(tracks, key=lambda t: t.observations[0].t):
        for index, earlier in enumerate(kept):
            if _same_item(earlier, track, rejoin_s):
                kept[index] = _joined(earlier, track)
                break
        else:
            kept.append(track)
    return tuple(kept)


def _same_item(earlier: Track, later: Track, rejoin_s: float) -> bool:
    boxes = {o.t: o.box for o in earlier.observations}
    shared = [(boxes[o.t], o.box) for o in later.observations if o.t in boxes]
    if shared:
        alike = sum(iou(a, b) >= SAME_OBJECT_IOU for a, b in shared)
        return alike * 2 >= len(shared)
    return _seen_again(earlier, later, rejoin_s)


# Two items pass through the same place on a belt, one after the other. So it counts as the same
# item only if, at the speed it went, it would have moved less than its own size in the gap.
def _seen_again(earlier: Track, later: Track, rejoin_s: float) -> bool:
    last, first = earlier.observations[-1], later.observations[0]
    gap = first.t - last.t
    if gap < 0 or gap > rejoin_s or _inside_share(last.box, first.box) < _SAME_PLACE:
        return False
    side = min(_side(last.box), _side(first.box))
    return max(_speed(earlier), _speed(later)) * gap <= side


def _joined(earlier: Track, later: Track) -> Track:
    best: dict[float, Observation] = {}
    for observation in (*earlier.observations, *later.observations):
        held = best.get(observation.t)
        if held is None or observation.score > held.score:
            best[observation.t] = observation
    return earlier.model_copy(update={"observations": tuple(best[t] for t in sorted(best))})


def _speed(track: Track) -> float:
    first, last = track.observations[0], track.observations[-1]
    if last.t <= first.t:
        return 0.0
    across = (last.box.x0 + last.box.x1 - first.box.x0 - first.box.x1) / 2
    down = (last.box.y0 + last.box.y1 - first.box.y0 - first.box.y1) / 2
    return (across**2 + down**2) ** 0.5 / (last.t - first.t)


def iou(a: Box, b: Box) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _inside_share(a: Box, b: Box) -> float:
    smaller = min(_area(a), _area(b))
    return _intersection(a, b) / smaller if smaller > 0 else 0.0


def _intersection(a: Box, b: Box) -> float:
    width = min(a.x1, b.x1) - max(a.x0, b.x0)
    height = min(a.y1, b.y1) - max(a.y0, b.y0)
    return width * height if width > 0 and height > 0 else 0.0


def _area(box: Box) -> float:
    return (box.x1 - box.x0) * (box.y1 - box.y0)


def _side(box: Box) -> float:
    return min(box.x1 - box.x0, box.y1 - box.y0)
