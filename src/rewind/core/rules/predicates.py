from collections import Counter
from collections.abc import Callable
from typing import Any

from rewind.contracts import CountAtLeast, Has, InZone, TimeSpan
from rewind.core.rules.items import Item
from rewind.core.rules.zones import area, center, contains

# Returns why the predicate fails for the item within the span, or None when it holds.
type Evaluator = Callable[[Any, Item, TimeSpan], str | None]

# Keyed by the predicate's `kind`; a new predicate is one new entry, never a change to dispatch.
PREDICATES: dict[str, Evaluator] = {}


def evaluates[E: Evaluator](kind: str) -> Callable[[E], E]:
    def register(evaluator: E) -> E:
        PREDICATES[kind] = evaluator
        return evaluator

    return register


@evaluates("has")
def has(predicate: Has, item: Item, span: TimeSpan) -> str | None:
    if item.parts_in(predicate.part, span):
        return None
    return f"no {item.part_phrases[predicate.part]}"


@evaluates("count_at_least")
def count_at_least(predicate: CountAtLeast, item: Item, span: TimeSpan) -> str | None:
    # Counted within one frame, so the same part seen in many frames is not counted twice.
    per_frame = Counter(o.t for o in item.parts_in(predicate.part, span))
    most = max(per_frame.values(), default=0)
    if most >= predicate.n:
        return None
    return f"only {most} of {predicate.n} {item.part_phrases[predicate.part]}"


@evaluates("in_zone")
def in_zone(predicate: InZone, item: Item, span: TimeSpan) -> str | None:
    zone = area(item.zones, predicate.zone)
    if any(contains(zone, center(o.box)) for o in item.seen_in(span)):
        return None
    return f"not in {predicate.zone}"
