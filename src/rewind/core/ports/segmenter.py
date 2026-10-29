from dataclasses import dataclass
from typing import Literal, Protocol

from rewind.contracts import Box, Charge, Observation, Point


@dataclass(frozen=True, slots=True)
class ConceptPrompt:
    # Exactly one noun phrase; providers find nothing for lists.
    text: str


@dataclass(frozen=True, slots=True)
class PromptPoint:
    point: Point
    positive: bool = True


@dataclass(frozen=True, slots=True)
class ObjectPrompt:
    # Footage time of the frame the object is marked on.
    t: float
    object_id: int
    target: Box | tuple[PromptPoint, ...]


@dataclass(frozen=True, slots=True)
class ObjectPrompts:
    # The concept the resulting observations are labelled with.
    concept: str
    objects: tuple[ObjectPrompt, ...]
    # Tracking forward from a tripwire frame bills only the tracked range.
    direction: Literal["both", "forward", "backward"] = "both"


Prompt = ConceptPrompt | ObjectPrompts


@dataclass(frozen=True, slots=True)
class SegmentRequest:
    footage_id: str
    # A proxy clip already at the planned fps, or one still image: every frame sent is billed.
    media_key: str
    width: int
    height: int
    # Footage time of each frame in the media, so results map back to the original footage.
    frame_times: tuple[float, ...]
    prompt: Prompt


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

    @property
    def max_prompt_objects(self) -> int: ...

    @property
    def max_tracked_objects(self) -> int | None:
        """Most objects one request follows; once it has found this many, it finds no more."""
        ...

    @property
    def min_billed_frames(self) -> int:
        """Every request bills at least this many frames, so join short windows into one clip."""
        ...

    def estimate(self, request: SegmentRequest) -> tuple[Charge, ...]:
        """Prices the frames sent, or the tracked range, at no less than min_billed_frames."""
        ...

    async def segment(self, request: SegmentRequest) -> Segmentation:
        """Finds and tracks one concept; charges come from the provider's reported usage."""
        ...
