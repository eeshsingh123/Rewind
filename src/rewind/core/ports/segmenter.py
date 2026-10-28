from dataclasses import dataclass
from typing import Protocol

from rewind.contracts import Charge, Observation


@dataclass(frozen=True, slots=True)
class SegmentRequest:
    footage_id: str
    # A proxy clip already at the planned fps, or one still image: every frame sent is billed.
    media_key: str
    # Footage time of each frame in the media, so results map back to the original footage.
    frame_times: tuple[float, ...]
    concept: str


@dataclass(frozen=True, slots=True)
class Segmentation:
    observations: tuple[Observation, ...]
    charges: tuple[Charge, ...]


class SegmenterError(Exception):
    pass


class Segmenter(Protocol):
    @property
    def model_version(self) -> str: ...

    @property
    def max_frames(self) -> int: ...

    def estimate(self, request: SegmentRequest) -> tuple[Charge, ...]: ...

    async def segment(self, request: SegmentRequest) -> Segmentation:
        """Finds and tracks one concept; charges come from the provider's reported usage."""
        ...
