from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from rewind.contracts.base import Concept, Contract, Key


class Entity(Contract):
    key: Key
    phrase: Concept


class EntersZone(Contract):
    kind: Literal["enters"] = "enters"
    zone: Key


class ExitsView(Contract):
    kind: Literal["exits_view"] = "exits_view"


class EndOfFootage(Contract):
    kind: Literal["end_of_footage"] = "end_of_footage"


Checkpoint = Annotated[EntersZone | ExitsView | EndOfFootage, Field(discriminator="kind")]


class Has(Contract):
    kind: Literal["has"] = "has"
    part: Key


class InZone(Contract):
    kind: Literal["in_zone"] = "in_zone"
    zone: Key


class CountAtLeast(Contract):
    kind: Literal["count_at_least"] = "count_at_least"
    part: Key
    n: int = Field(ge=1)


Predicate = Annotated[Has | InZone | CountAtLeast, Field(discriminator="kind")]


class Requirement(Contract):
    predicate: Predicate
    timing: Literal["at_checkpoint", "ever_before"]


class Before(Contract):
    first: Key
    then: Key


class Dwell(Contract):
    zone: Key
    max_s: float = Field(gt=0)


class Rate(Contract):
    min_per_minute: float = Field(gt=0)


class RuleSpec(Contract):
    entity: Entity
    parts: dict[Key, Concept] = Field(default_factory=dict[Key, Concept])
    checkpoint: Checkpoint
    requirements: tuple[Requirement, ...] = ()
    order: tuple[Before, ...] = ()
    dwell: Dwell | None = None
    rate: Rate | None = None
    unsupported: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _parts_declared(self) -> Self:
        used = [r.predicate.part for r in self.requirements if not isinstance(r.predicate, InZone)]
        used += [key for step in self.order for key in (step.first, step.then)]
        missing = sorted(set(used) - self.parts.keys())
        if missing:
            raise ValueError(f"parts used but not declared: {', '.join(missing)}")
        if self.entity.key in self.parts:
            raise ValueError("the entity key must not also be a part key")
        return self
