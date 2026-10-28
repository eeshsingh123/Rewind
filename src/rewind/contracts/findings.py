from typing import Literal

from pydantic import Field

from rewind.contracts.base import Contract, Id, Key, Seconds, TimeSpan

EventKind = Literal["enters_zone", "exits_view", "dwell", "jam"]


class Event(Contract):
    id: Id
    footage_id: Id
    kind: EventKind
    span: TimeSpan
    track_id: Id | None = None
    zone: Key | None = None


class Violation(Contract):
    id: Id
    rule_id: Id
    footage_id: Id
    # None for rules about the whole line rather than one item, such as a minimum rate.
    track_id: Id | None
    t: Seconds
    reason: str = Field(min_length=1)
    clip: TimeSpan
    order_id: Id | None = None


class Insight(Contract):
    id: Id
    footage_id: Id
    kind: Key
    summary: str = Field(min_length=1)
    event_ids: tuple[Id, ...] = Field(min_length=1)
