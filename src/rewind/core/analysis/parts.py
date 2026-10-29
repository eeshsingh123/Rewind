import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rewind.contracts import Charge, Footage, InZone, Observation, RuleSpec
from rewind.core.perception import (
    CachedFrames,
    CacheKey,
    ObservationStore,
    OnProgress,
    frame_index,
)
from rewind.core.ports import BlobStore, ConceptPrompt, Segmenter, SegmentRequest
from rewind.core.rules import Check


class StillGrabber(Protocol):
    async def still(self, source: Path, t: float, destination: Path) -> None:
        """Writes the frame at footage time t as a JPEG."""
        ...


@dataclass(frozen=True, slots=True)
class PartLooks:
    observations: tuple[Observation, ...]
    charges: tuple[Charge, ...]


# Parts a rule needs only at its checkpoint are looked for on one still per item, not tracked:
# a video request follows at most 16 objects, and a busy line has more labels than that in one
# frame. Parts needed over time (ever_before, order) are still tracked on video.
def checkpoint_parts(spec: RuleSpec) -> frozenset[str]:
    over_time = {key for step in spec.order for key in (step.first, step.then)}
    at_checkpoint: set[str] = set()
    for requirement in spec.requirements:
        if isinstance(requirement.predicate, InZone):
            continue
        if requirement.timing == "ever_before":
            over_time.add(requirement.predicate.part)
        else:
            at_checkpoint.add(requirement.predicate.part)
    return frozenset(spec.parts[key] for key in at_checkpoint - over_time)


# What video has to follow for a rule: the item, and the parts not left to stills.
def tracking_spec(spec: RuleSpec) -> RuleSpec:
    stills = checkpoint_parts(spec)
    tracked = {key: phrase for key, phrase in spec.parts.items() if phrase not in stills}
    return RuleSpec(entity=spec.entity, parts=tracked, checkpoint=spec.checkpoint)


def still_times(spec: RuleSpec, checks: Iterable[Check]) -> dict[str, set[float]]:
    times: defaultdict[str, set[float]] = defaultdict(set)
    for phrase in checkpoint_parts(spec):
        times[phrase].update(check.look_at for check in checks)
    return times


class PartLooker:
    def __init__(self, blobs: BlobStore, store: ObservationStore, stills: StillGrabber) -> None:
        self._blobs = blobs
        self._store = store
        self._stills = stills

    # One image call per concept and moment, cached like video frames, so a rerun is free.
    async def look(
        self,
        footage: Footage,
        times: Mapping[str, set[float]],
        segmenter: Segmenter,
        on_progress: OnProgress | None = None,
    ) -> PartLooks:
        observations: list[Observation] = []
        charges: list[Charge] = []
        total = sum(len(wanted) for wanted in times.values())
        done = 0
        for concept, wanted in sorted(times.items()):
            # Kept apart from video frames of the same concept: a still is not a tracked frame.
            key = CacheKey(footage.content_hash, concept, f"{segmenter.model_version}/still")
            frames = dict((await self._store.load(key, footage.id)).frames)
            for t in sorted(wanted):
                index = frame_index(t, footage.fps)
                if index not in frames:
                    found, paid = await self._segment(footage, concept, t, segmenter)
                    frames[index] = found
                    charges.extend(paid)
                observations.extend(frames[index])
                done += 1
                if on_progress is not None:
                    await on_progress(done, total)
            await self._store.save(key, CachedFrames(frames=frames))
        return PartLooks(tuple(observations), tuple(charges))

    async def _segment(
        self, footage: Footage, concept: str, t: float, segmenter: Segmenter
    ) -> tuple[tuple[Observation, ...], Sequence[Charge]]:
        image_key = f"stills/{footage.id}/{frame_index(t, footage.fps)}.jpg"
        if not await self._blobs.exists(image_key):
            source = await self._blobs.local_path(footage.original_key)
            with tempfile.TemporaryDirectory() as workdir:
                destination = Path(workdir) / "still.jpg"
                await self._stills.still(source, t, destination)
                await self._blobs.write_file(image_key, destination)
        request = SegmentRequest(
            footage.id, image_key, footage.width, footage.height, (t,), ConceptPrompt(concept)
        )
        result = await segmenter.segment(request)
        return result.observations, result.charges
