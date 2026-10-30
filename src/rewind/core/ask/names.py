from collections.abc import Mapping, Sequence

from pydantic import BaseModel, JsonValue, ValidationError

from rewind.contracts import Observation
from rewind.core.perception import SAME_OBJECT_IOU, iou

# Names Muse may suggest; the words asked are tried as well.
MAX_NAMES = 3
# Each name tracked is a full pass over the video.
MAX_TRACKED_NAMES = 2
# A video request returns nothing below this, so a still's weaker finds would not be tracked.
TRACKED_SCORE = 0.7

NAMING = (
    "You help a visual search find things in footage. It takes one short everyday name at a "
    "time and matches how things look, so one name can miss things of the same kind that look "
    "different."
)
NAME = (
    "A person asks about every {thing}. From these frames, give up to {most} short names of one "
    "to three words that together cover every look such things have here, such as hard, soft, "
    "wrapped or bagged. No places, no conditions, and nothing a person would not count as a "
    "{thing}."
)
NAMES_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["names"],
    "properties": {"names": {"type": "array", "items": {"type": "string"}}},
}


class _Names(BaseModel):
    names: list[str]


# The words asked come first; what Muse answered is only ever an addition to them.
def names_from(text: str, thing: str) -> tuple[str, ...]:
    try:
        suggested = _Names.model_validate_json(text).names
    except ValidationError:
        return (thing,)
    cleaned = (" ".join(name.lower().split()) for name in suggested[:MAX_NAMES])
    return tuple(dict.fromkeys((thing, *(name for name in cleaned if name))))


# The name that adds the most things on these frames, then the next, until one adds nothing. A
# name that fills a request on one frame loses what it cannot follow, so it is the last resort.
def chosen_names(
    names: Sequence[str], looks: Mapping[str, Sequence[Observation]], limit: int | None
) -> tuple[str, ...]:
    sure = {n: [o for o in looks.get(n, ()) if o.score >= TRACKED_SCORE] for n in names}
    usable = [n for n in names if not _fills(sure[n], limit)] or list(names)
    chosen: list[str] = []
    while len(chosen) < MAX_TRACKED_NAMES:
        covered = [o for name in chosen for o in sure[name]]
        added = {n: _new(sure[n], covered) for n in usable if n not in chosen}
        best = max(added, key=lambda n: added[n], default=None)
        if best is None or added[best] == 0:
            break
        chosen.append(best)
    return tuple(chosen) or (names[0],)


def _fills(seen: Sequence[Observation], limit: int | None) -> bool:
    if limit is None:
        return False
    return any(sum(o.t == other.t for o in seen) >= limit for other in seen)


def _new(seen: Sequence[Observation], covered: Sequence[Observation]) -> int:
    return sum(
        not any(o.t == c.t and iou(o.box, c.box) >= SAME_OBJECT_IOU for c in covered) for o in seen
    )
