import random
from collections.abc import Callable
from dataclasses import dataclass

from rewind.contracts.scenario import PropValue, Vector3

# Mirrors the rule engine: an item last seen this close to the end has not left the view.
EXIT_MARGIN_S = 1.0


@dataclass(frozen=True, slots=True)
class Part:
    key: str
    # The fault that makes the station skip this part.
    fault: str


@dataclass(frozen=True, slots=True)
class Station:
    # Also the key of the zone the station occupies, from x = start to x = end along the belt.
    key: str
    start: float
    end: float
    parts: tuple[Part, ...] = ()

    @property
    def x(self) -> float:
        return (self.start + self.end) / 2


@dataclass(frozen=True, slots=True)
class ItemTimeline:
    first_seen: float
    last_seen: float
    exits: bool
    zone_entry: dict[str, float]
    zone_stay: dict[str, float]
    # When each part appears on the item; None when a fault skipped it.
    parts: dict[str, float | None]
    faults: frozenset[str]

    def has(self, part: str, t: float) -> bool:
        appears = self.parts.get(part)
        return appears is not None and appears <= t


Judge = Callable[[ItemTimeline], tuple[str, ...]]
Styler = Callable[[random.Random], dict[str, PropValue]]


@dataclass(frozen=True, slots=True)
class Line:
    item_kind: str
    # Extent along the belt, across it, and height, in metres.
    item_size: Vector3
    belt_top: float
    belt_width: float
    stations: tuple[Station, ...]
    judge: Judge
    style: Styler

    @property
    def item_center_z(self) -> float:
        return self.belt_top + self.item_size[2] / 2

    @property
    def faults(self) -> frozenset[str]:
        return frozenset(p.fault for s in self.stations for p in s.parts)
