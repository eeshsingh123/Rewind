from typing import Literal

from pydantic import Field

from rewind.contracts.base import Contract, Id, Key, Seconds, TimeSpan
from rewind.contracts.geometry import Zone

Vector3 = tuple[float, float, float]
PropValue = str | int | float | bool


class Camera(Contract):
    position: Vector3
    look_at: Vector3
    focal_mm: float = Field(gt=0)


class ScenarioConfig(Contract):
    name: Key
    builder: Key
    seed: int
    duration_s: float = Field(gt=0)
    fps: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    item_interval_s: float = Field(gt=0)
    camera: Camera
    # Fault key (e.g. missing_tape) to its probability per item.
    faults: dict[Key, float] = Field(default_factory=dict[Key, float])
    jams: tuple[TimeSpan, ...] = ()
    # Settings that only one builder understands.
    params: dict[Key, PropValue] = Field(default_factory=dict[Key, PropValue])


class GroundTruthItem(Contract):
    id: Id
    order_id: Id | None = None
    t_spawn: Seconds
    t_cross: dict[Key, Seconds] = Field(default_factory=dict[Key, Seconds])
    props: dict[Key, PropValue] = Field(default_factory=dict[Key, PropValue])
    expected_violations: tuple[Key, ...] = ()


class ScenarioEvent(Contract):
    kind: Literal["jam"]
    span: TimeSpan


class GroundTruth(Contract):
    scenario: Key
    seed: int
    duration_s: float = Field(gt=0)
    fps: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    camera: Camera
    zones: tuple[Zone, ...]
    items: tuple[GroundTruthItem, ...]
    events: tuple[ScenarioEvent, ...] = ()
