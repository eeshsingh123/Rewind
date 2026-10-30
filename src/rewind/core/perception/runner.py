import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from rewind.contracts import Box, Charge, Footage, Observation, SamplingPlan, TimeSpan
from rewind.core.perception.cache import CachedFrames, CacheKey, ObservationStore, frame_index
from rewind.core.perception.proxy import ProxyClip, ProxyMaker
from rewind.core.ports import (
    ConceptPrompt,
    ObjectPrompt,
    ObjectPrompts,
    Segmenter,
    SegmentRequest,
)

logger = logging.getLogger(__name__)

# Spans are cut a little short of the frame limit, as a span's frame count can round up.
_FRAME_HEADROOM = 0.98
# Boxes of the same object on the same frame, seen by two requests, overlap at least this much.
_SAME_OBJECT_IOU = 0.5

# Called with (done, total) after each piece of work, so a chat can show how far a check is.
type OnProgress = Callable[[int, int], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class PerceptionResult:
    observations: tuple[Observation, ...]
    charges: tuple[Charge, ...]
    paid_requests: int
    cached_requests: int


class PerceptionRunner:
    def __init__(self, proxies: ProxyMaker, store: ObservationStore) -> None:
        self._proxies = proxies
        self._store = store

    # One request per concept and fps over all its windows joined, unless the cache has it.
    async def run(
        self,
        footage: Footage,
        plan: SamplingPlan,
        segmenter: Segmenter,
        on_progress: OnProgress | None = None,
    ) -> PerceptionResult:
        observations: list[Observation] = []
        charges: list[Charge] = []
        paid = cached = 0
        work = [
            (concept, fps, chunk)
            for (concept, fps), spans in _group_windows(plan).items()
            for chunk in _chunks(spans, fps, segmenter.max_frames)
        ]
        for done, (concept, fps, chunk) in enumerate(work, 1):
            clip = await self._proxies.ensure(
                footage.content_hash, footage.original_key, chunk, fps
            )
            found, new_charges = await self._segment(footage, clip, concept, segmenter)
            observations.extend(found)
            charges.extend(new_charges)
            if new_charges:
                paid += 1
            else:
                cached += 1
            if on_progress is not None:
                await on_progress(done, len(work))
        return PerceptionResult(
            observations=tuple(observations),
            charges=tuple(charges),
            paid_requests=paid,
            cached_requests=cached,
        )

    # Follows the one thing under a box through the whole video, from that frame both ways. The
    # whole video, since what is marked may stand still where nothing else moves.
    async def follow(
        self, footage: Footage, t: float, box: Box, fps: float, segmenter: Segmenter, concept: str
    ) -> PerceptionResult:
        marked = f"follow {t:.2f} {box.x0:.3f} {box.y0:.3f} {box.x1:.3f} {box.y1:.3f}"
        key = CacheKey(footage.content_hash, marked, segmenter.model_version)
        stored = await self._store.load(key, footage.id)
        if stored.frames:
            seen = tuple(o for found in stored.frames.values() for o in found)
            return PerceptionResult(seen, (), paid_requests=0, cached_requests=1)
        whole = TimeSpan(start=0, end=footage.duration_s)
        clip = await self._proxies.ensure(footage.content_hash, footage.original_key, [whole], fps)
        request = SegmentRequest(
            footage_id=footage.id,
            media_key=clip.proxy.blob_key,
            width=clip.width,
            height=clip.height,
            frame_times=clip.frame_times,
            prompt=ObjectPrompts(concept, (ObjectPrompt(t, 0, box),)),
        )
        result = await segmenter.segment(request)
        frames: dict[int, tuple[Observation, ...]] = {
            frame_index(frame_t, footage.fps): () for frame_t in clip.frame_times
        }
        found: defaultdict[int, list[Observation]] = defaultdict(list)
        for observation in result.observations:
            found[frame_index(observation.t, footage.fps)].append(observation)
        frames.update({index: tuple(seen) for index, seen in found.items()})
        await self._store.save(key, CachedFrames(frames=frames))
        return PerceptionResult(result.observations, result.charges, 1, 0)

    async def _segment(
        self, footage: Footage, clip: ProxyClip, concept: str, segmenter: Segmenter
    ) -> tuple[list[Observation], tuple[Charge, ...]]:
        key = CacheKey(footage.content_hash, concept, segmenter.model_version)
        stored = await self._store.load(key, footage.id)
        indices = [frame_index(t, footage.fps) for t in clip.frame_times]
        hits = [o for index in dict.fromkeys(indices) for o in stored.frames.get(index, ())]
        missing = [index not in stored.frames for index in indices]
        if not any(missing):
            return hits, ()
        if not all(missing):
            # Only the frames not seen yet are sent: the rest would be paid for twice.
            spans = _spans_around(clip.frame_times, missing, clip.proxy.fps)
            clip = await self._proxies.ensure(
                footage.content_hash, footage.original_key, spans, clip.proxy.fps
            )
        observations, charges = await self._track(footage, clip, concept, segmenter)
        found: defaultdict[int, list[Observation]] = defaultdict(list)
        for observation in observations:
            found[frame_index(observation.t, footage.fps)].append(observation)
        frames = dict(stored.frames)
        frames.update({frame_index(t, footage.fps): () for t in clip.frame_times})
        frames.update({index: tuple(seen) for index, seen in found.items()})
        await self._store.save(key, CachedFrames(frames=frames))
        return hits + observations, charges

    # A request stops finding new objects once it has found as many as the segmenter can follow,
    # so a busy line would lose every later item. From the frame where it filled up, a new request
    # takes over with fresh slots; its ids are matched to the old ones on that shared frame.
    async def _track(
        self, footage: Footage, clip: ProxyClip, concept: str, segmenter: Segmenter
    ) -> tuple[list[Observation], tuple[Charge, ...]]:
        kept: list[Observation] = []
        charges: list[Charge] = []
        previous: list[Observation] = []
        next_id = 0
        while True:
            request = SegmentRequest(
                footage_id=footage.id,
                media_key=clip.proxy.blob_key,
                width=clip.width,
                height=clip.height,
                frame_times=clip.frame_times,
                prompt=ConceptPrompt(concept),
            )
            result = await segmenter.segment(request)
            charges.extend(result.charges)
            first = frame_index(clip.frame_times[0], footage.fps)
            found, next_id = _continue_ids(previous, result.observations, footage.fps, next_id)
            full_at = _full_at(result.observations, segmenter.max_tracked_objects)
            if full_at is None or frame_index(full_at, footage.fps) <= first:
                if full_at is not None:
                    logger.warning("%s: too many %r in one frame to follow", footage.id, concept)
                return kept + found, tuple(charges)
            full = frame_index(full_at, footage.fps)
            kept.extend(o for o in found if frame_index(o.t, footage.fps) < full)
            previous = [o for o in found if frame_index(o.t, footage.fps) == full]
            rest = [frame_index(t, footage.fps) >= full for t in clip.frame_times]
            clip = await self._proxies.ensure(
                footage.content_hash,
                footage.original_key,
                _spans_around(clip.frame_times, rest, clip.proxy.fps),
                clip.proxy.fps,
            )


# When the request found its last new object, if that filled every slot it has.
def _full_at(observations: Sequence[Observation], limit: int | None) -> float | None:
    first_seen: dict[int, float] = {}
    for observation in sorted(observations, key=lambda o: o.t):
        if observation.object_id is not None:
            first_seen.setdefault(observation.object_id, observation.t)
    if limit is None or len(first_seen) < limit:
        return None
    return max(first_seen.values())


# Renumbers a request's objects from next_id up. After a full request, `previous` holds what it
# saw on the frame this request starts on: an object found there again keeps its earlier id.
def _continue_ids(
    previous: Sequence[Observation],
    found: Sequence[Observation],
    footage_fps: float,
    next_id: int,
) -> tuple[list[Observation], int]:
    renamed: dict[int, int] = {}
    if previous:
        shared = frame_index(previous[0].t, footage_fps)
        starts = [o for o in found if frame_index(o.t, footage_fps) == shared]
        pairs = sorted(
            ((_iou(old.box, new.box), old.object_id, new.object_id)
             for old in previous for new in starts),
            key=lambda pair: -pair[0],
        )  # fmt: skip
        for iou, old_id, new_id in pairs:
            if iou < _SAME_OBJECT_IOU:
                break
            if old_id is None or new_id is None or new_id in renamed:
                continue
            if old_id not in renamed.values():
                renamed[new_id] = old_id
    for object_id in sorted({o.object_id for o in found if o.object_id is not None}):
        if object_id not in renamed:
            renamed[object_id] = next_id
            next_id += 1
    renumbered = [
        o if o.object_id is None else o.model_copy(update={"object_id": renamed[o.object_id]})
        for o in found
    ]
    return renumbered, next_id


def _iou(a: Box, b: Box) -> float:
    width = min(a.x1, b.x1) - max(a.x0, b.x0)
    height = min(a.y1, b.y1) - max(a.y0, b.y0)
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    union = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter
    return inter / union


def _group_windows(plan: SamplingPlan) -> dict[tuple[str, float], list[TimeSpan]]:
    groups: defaultdict[tuple[str, float], list[TimeSpan]] = defaultdict(list)
    for window in plan.windows:
        for concept in window.concepts:
            groups[(concept, window.fps)].append(window.span)
    return {key: _merge(spans) for key, spans in groups.items()}


def _merge(spans: Sequence[TimeSpan]) -> list[TimeSpan]:
    merged: list[TimeSpan] = []
    for span in sorted(spans, key=lambda s: s.start):
        if merged and span.start <= merged[-1].end:
            merged[-1] = TimeSpan(start=merged[-1].start, end=max(merged[-1].end, span.end))
        else:
            merged.append(span)
    return merged


def _chunks(spans: Sequence[TimeSpan], fps: float, max_frames: int) -> list[list[TimeSpan]]:
    limit_s = max_frames * _FRAME_HEADROOM / fps
    chunks: list[list[TimeSpan]] = [[]]
    used = 0.0
    for span in spans:
        start = span.start
        while start < span.end:
            if used >= limit_s:
                chunks.append([])
                used = 0.0
            end = min(span.end, start + limit_s - used)
            chunks[-1].append(TimeSpan(start=start, end=end))
            used += end - start
            start = end
    return [chunk for chunk in chunks if chunk]


# A span around each run of adjacent flagged frames, narrow enough to hold no other frame.
def _spans_around(
    frame_times: Sequence[float], flagged: Sequence[bool], fps: float
) -> list[TimeSpan]:
    margin = 0.4 / fps
    spans: list[TimeSpan] = []
    run_start: float | None = None
    for position, (t, flag) in enumerate(zip(frame_times, flagged, strict=True)):
        if flag and run_start is None:
            run_start = t
        continues = (
            position + 1 < len(flagged)
            and flagged[position + 1]
            and frame_times[position + 1] - t < 1.5 / fps
        )
        last_of_run = flag and not continues
        if last_of_run and run_start is not None:
            spans.append(TimeSpan(start=max(0.0, run_start - margin), end=t + margin))
            run_start = None
    return spans
