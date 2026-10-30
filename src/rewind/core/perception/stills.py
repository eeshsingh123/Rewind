import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rewind.contracts import Charge, Footage, Observation
from rewind.core.perception.cache import CachedFrames, CacheKey, ObservationStore, frame_index
from rewind.core.ports import BlobStore, ConceptPrompt, Segmenter, SegmentRequest


# Where the still of one frame is kept. A photo is stored here too, as its own first frame.
def still_key(footage_id: str, frame: int) -> str:
    return f"stills/{footage_id}/{frame}.jpg"


class StillGrabber(Protocol):
    async def still(self, source: Path, t: float, destination: Path) -> None:
        """Writes the frame at footage time t as a JPEG."""
        ...


@dataclass(frozen=True, slots=True)
class PartLooks:
    observations: tuple[Observation, ...]
    charges: tuple[Charge, ...]


class PartLooker:
    def __init__(self, blobs: BlobStore, store: ObservationStore, stills: StillGrabber) -> None:
        self._blobs = blobs
        self._store = store
        self._stills = stills

    # One image call per concept and moment, cached like video frames, so a rerun is free.
    async def look(
        self, footage: Footage, times: Mapping[str, set[float]], segmenter: Segmenter
    ) -> PartLooks:
        observations: list[Observation] = []
        charges: list[Charge] = []
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
            await self._store.save(key, CachedFrames(frames=frames))
        return PartLooks(tuple(observations), tuple(charges))

    async def still(self, footage: Footage, t: float) -> str:
        image_key = still_key(footage.id, frame_index(t, footage.fps))
        if not await self._blobs.exists(image_key):
            source = await self._blobs.local_path(footage.original_key)
            with tempfile.TemporaryDirectory() as workdir:
                destination = Path(workdir) / "still.jpg"
                await self._stills.still(source, t, destination)
                await self._blobs.write_file(image_key, destination)
        return image_key

    async def _segment(
        self, footage: Footage, concept: str, t: float, segmenter: Segmenter
    ) -> tuple[tuple[Observation, ...], Sequence[Charge]]:
        image_key = await self.still(footage, t)
        request = SegmentRequest(
            footage.id, image_key, footage.width, footage.height, (t,), ConceptPrompt(concept)
        )
        result = await segmenter.segment(request)
        return result.observations, result.charges
