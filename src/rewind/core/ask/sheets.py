from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rewind.contracts import Box


@dataclass(frozen=True, slots=True)
class Tile:
    # A moment of the footage shown to the model, with the item drawn and a label on it.
    t: float
    box: Box | None
    label: str


class SheetMaker(Protocol):
    async def make_sheet(self, source: Path, destination: Path, tiles: Sequence[Tile]) -> None:
        """One JPEG of the footage at each tile's time, its box drawn and its label written."""
        ...
