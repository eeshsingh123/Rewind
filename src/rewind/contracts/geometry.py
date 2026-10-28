from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from rewind.contracts.base import Concept, Contract, Key

Unit = Annotated[float, Field(ge=0, le=1)]


class Point(Contract):
    x: Unit
    y: Unit


class Box(Contract):
    # Normalized to the frame, so boxes and zones survive a change of resolution.
    x0: Unit
    y0: Unit
    x1: Unit
    y1: Unit

    @model_validator(mode="after")
    def _corners_ordered(self) -> Self:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("box corners must satisfy x0 <= x1 and y0 <= y1")
        return self


class RleMask(Contract):
    # Covers only the crop under the box: row-major runs, alternating, starting with 0s.
    height: int = Field(gt=0)
    width: int = Field(gt=0)
    counts: tuple[int, ...]

    @model_validator(mode="after")
    def _runs_cover_crop(self) -> Self:
        if any(run < 0 for run in self.counts) or sum(self.counts) != self.height * self.width:
            raise ValueError("mask runs must be non-negative and cover height * width")
        return self


class Polygon(Contract):
    kind: Literal["polygon"] = "polygon"
    points: tuple[Point, ...] = Field(min_length=3)


class Tripwire(Contract):
    kind: Literal["tripwire"] = "tripwire"
    start: Point
    end: Point


class ViewExit(Contract):
    kind: Literal["view_exit"] = "view_exit"


Shape = Annotated[Polygon | Tripwire | ViewExit, Field(discriminator="kind")]


class Zone(Contract):
    key: Key
    shape: Shape
    source: Literal["drawn", "text", "implicit"]
    phrase: Concept | None = None
