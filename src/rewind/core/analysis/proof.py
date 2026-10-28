import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from rewind.contracts import Box, TimeSpan, Track, Violation
from rewind.core.rules import Check, Item

# A part found on an item is drawn this long either side of the moment it was seen.
PART_SHOWN_S = 0.6
_MOMENT = re.compile(r" (when|before) (it left the view|the footage ended|it entered \S+)$")


class Sighting(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    t: float
    box: Box


class Mark(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    # "flagged" is the item the clip is about; others in view are drawn thin, by their result.
    kind: Literal["flagged", "failed", "passed"]
    tag: str
    path: tuple[Sighting, ...]
    # Parts found on the item, as boxes relative to the item's own box, so they move with it.
    parts: tuple[Sighting, ...] = ()


class ProofMarks(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    marks: tuple[Mark, ...]


class ClipMaker(Protocol):
    async def make_clip(
        self, source: Path, destination: Path, span: TimeSpan, marks: Sequence[Mark]
    ) -> None:
        """Cuts the span and draws each mark's box, moving between its sightings."""
        ...


# "no label when it left the view; tape came before flap" -> "no label · tape came before flap"
def short_reason(reason: str) -> str:
    return " · ".join(_MOMENT.sub("", part) for part in reason.split("; "))


# The flagged item, and every other item the rule judged that is in view during the clip.
def proof_marks(
    flagged: Violation,
    items: Mapping[str, Item],
    checks: Sequence[Check],
    failed: Mapping[str, Violation],
) -> ProofMarks:
    marks: list[Mark] = []
    judged = [check.track_id for check in checks]
    shown = judged if flagged.track_id in judged else [*judged, flagged.track_id]
    for track_id in shown:
        item = items.get(track_id or "")
        if item is None:
            continue
        path = _within(item.track, flagged.clip)
        if not path:
            continue
        violation = failed.get(item.track.track_id)
        if track_id == flagged.track_id:
            kind, tag = "flagged", short_reason(flagged.reason)
        elif violation is not None:
            kind, tag = "failed", ""
        else:
            kind, tag = "passed", ""
        marks.append(Mark(kind=kind, tag=tag, path=path, parts=_parts(item, flagged.clip)))
    return ProofMarks(marks=tuple(marks))


def _within(track: Track, span: TimeSpan) -> tuple[Sighting, ...]:
    return tuple(
        Sighting(t=o.t, box=o.box)
        for o in sorted(track.observations, key=lambda o: o.t)
        if span.start - 1 <= o.t <= span.end + 1
    )


def _parts(item: Item, span: TimeSpan) -> tuple[Sighting, ...]:
    found: list[Sighting] = []
    for seen in item.parts.values():
        for part in seen:
            if not span.start - PART_SHOWN_S <= part.t <= span.end + PART_SHOWN_S:
                continue
            whole = min(item.path, key=lambda o: abs(o.t - part.t)).box
            found.append(Sighting(t=part.t, box=_relative(part.box, whole)))
    return tuple(sorted(found, key=lambda s: s.t))


def _relative(part: Box, whole: Box) -> Box:
    width, height = max(whole.x1 - whole.x0, 1e-6), max(whole.y1 - whole.y0, 1e-6)

    def x(value: float) -> float:
        return min(1.0, max(0.0, (value - whole.x0) / width))

    def y(value: float) -> float:
        return min(1.0, max(0.0, (value - whole.y0) / height))

    return Box(x0=x(part.x0), y0=y(part.y0), x1=x(part.x1), y1=y(part.y1))
