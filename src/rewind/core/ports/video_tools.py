from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rewind.contracts import TimeSpan


@dataclass(frozen=True, slots=True)
class VideoInfo:
    duration_s: float
    fps: float
    width: int
    height: int
    frame_count: int


class VideoToolsError(Exception):
    pass


class VideoTools(Protocol):
    async def probe(self, source: Path) -> VideoInfo: ...

    async def make_proxy(
        self,
        source: Path,
        destination: Path,
        spans: Sequence[TimeSpan],
        fps: float,
        max_height: int,
    ) -> tuple[float, ...]:
        """Joins the spans into one H.264 MP4 at fps; returns the footage time of each frame."""
        ...

    async def cut_clip(self, source: Path, destination: Path, span: TimeSpan) -> None: ...
