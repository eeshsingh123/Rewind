from pydantic import Field

from rewind.contracts.base import Concept, Contract, Id, Seconds, TimeSpan
from rewind.contracts.geometry import Box, RleMask


class Observation(Contract):
    footage_id: Id
    t: Seconds
    concept: Concept
    box: Box
    score: float = Field(ge=0, le=1)
    # The segmenter's own id: stable only while the object stays in view, within one call.
    object_id: int | None = None
    mask: RleMask | None = None


class Track(Contract):
    footage_id: Id
    track_id: Id
    concept: Concept
    observations: tuple[Observation, ...] = Field(min_length=1)


class SamplingWindow(Contract):
    span: TimeSpan
    fps: float = Field(gt=0)
    concepts: tuple[Concept, ...] = Field(min_length=1)


class SamplingPlan(Contract):
    footage_id: Id
    windows: tuple[SamplingWindow, ...]
