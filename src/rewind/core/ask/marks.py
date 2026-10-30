from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from rewind.contracts import Box, TimeSpan


class Sighting(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    t: float
    box: Box


class Mark(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    # "flagged" is the item the clip is about; "passed" is another item in view, drawn thin.
    # "area" is a part of the frame, drawn at its one sighting for the whole clip.
    kind: Literal["flagged", "passed", "area"]
    tag: str
    path: tuple[Sighting, ...]


class ProofMarks(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    marks: tuple[Mark, ...]


class ClipMaker(Protocol):
    async def make_clip(
        self, source: Path, destination: Path, span: TimeSpan, marks: Sequence[Mark]
    ) -> None:
        """Cuts the span and draws each mark's box, moving between its sightings."""
        ...
