from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from rewind.contracts import Charge, Footage, Observation, SamplingPlan, TimeSpan
from rewind.core.perception.cache import CachedFrames, CacheKey, ObservationStore, frame_index
from rewind.core.perception.proxy import ProxyClip, ProxyMaker
from rewind.core.ports import ConceptPrompt, Segmenter, SegmentRequest

# Spans are cut a little short of the frame limit, as a span's frame count can round up.
_FRAME_HEADROOM = 0.98


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
        self, footage: Footage, plan: SamplingPlan, segmenter: Segmenter
    ) -> PerceptionResult:
        observations: list[Observation] = []
        charges: list[Charge] = []
        paid = cached = 0
        for (concept, fps), spans in _group_windows(plan).items():
            for chunk in _chunks(spans, fps, segmenter.max_frames):
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
        return PerceptionResult(
            observations=tuple(observations),
            charges=tuple(charges),
            paid_requests=paid,
            cached_requests=cached,
        )

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
        request = SegmentRequest(
            footage_id=footage.id,
            media_key=clip.proxy.blob_key,
            width=clip.width,
            height=clip.height,
            frame_times=clip.frame_times,
            prompt=ConceptPrompt(concept),
        )
        result = await segmenter.segment(request)
        found: defaultdict[int, list[Observation]] = defaultdict(list)
        for observation in result.observations:
            found[frame_index(observation.t, footage.fps)].append(observation)
        frames = dict(stored.frames)
        frames.update({frame_index(t, footage.fps): () for t in clip.frame_times})
        frames.update({index: tuple(seen) for index, seen in found.items()})
        await self._store.save(key, CachedFrames(frames=frames))
        return hits + list(result.observations), result.charges


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
