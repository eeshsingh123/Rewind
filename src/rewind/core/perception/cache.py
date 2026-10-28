from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from rewind.contracts import Observation


@dataclass(frozen=True, slots=True)
class CacheKey:
    content_hash: str
    concept: str
    model_version: str


@dataclass(frozen=True, slots=True)
class CachedFrames:
    # Original-footage frame index -> what the segmenter saw there; empty when it found nothing.
    frames: Mapping[int, tuple[Observation, ...]]


class ObservationStore(Protocol):
    async def load(self, key: CacheKey, footage_id: str) -> CachedFrames:
        """Every frame stored for this content, concept and model, labelled with footage_id."""
        ...

    async def save(self, key: CacheKey, cached: CachedFrames) -> None:
        """Replaces what is stored for the key."""
        ...


def frame_index(t: float, footage_fps: float) -> int:
    return round(t * footage_fps)
