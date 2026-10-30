import hashlib
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from rewind.contracts import Footage, TimeSpan
from rewind.core.ask.evidence import Evidence
from rewind.core.ask.marks import ClipMaker, Mark, ProofMarks
from rewind.core.ask.session import Session
from rewind.core.ports import BlobStore, VideoTools

# A clip longer than this would be too big for a chat; a longer stretch shows its start.
MAX_CLIP_S = 20.0
MIN_CLIP_S = 3.0


class ProofMaker(ClipMaker, Protocol):
    async def make_photo(self, source: Path, destination: Path, marks: Sequence[Mark]) -> None:
        """Writes the photo as a JPEG with each mark's box drawn."""
        ...


# The proof behind a claim, made once: its evidence's stretch as a clip with the boxes drawn, or
# on a photo, the photo with the boxes drawn. The key ends in .mp4 or .jpg.
class Proofs:
    def __init__(self, blobs: BlobStore, video: VideoTools, maker: ProofMaker) -> None:
        self._blobs = blobs
        self._video = video
        self._maker = maker

    async def render(self, session: Session, evidence: Evidence) -> str | None:
        if evidence.proof_key is not None:
            return evidence.proof_key
        if evidence.clip is None:
            return None
        if session.photo:
            return await self._photo(session.footage, evidence)
        return await self._clip(session.footage, evidence.clip, evidence)

    async def _photo(self, footage: Footage, evidence: Evidence) -> str:
        if not evidence.marks:
            return footage.original_key
        marks = ProofMarks(marks=evidence.marks)
        key = f"ask/proofs/{_digest(f'{footage.content_hash}|{marks.model_dump_json()}')}.jpg"
        if await self._blobs.exists(key):
            return key
        source = await self._blobs.local_path(footage.original_key)
        with tempfile.TemporaryDirectory() as workdir:
            destination = Path(workdir) / "proof.jpg"
            await self._maker.make_photo(source, destination, marks.marks)
            await self._blobs.write_file(key, destination)
        return key

    async def _clip(self, footage: Footage, clip: TimeSpan, evidence: Evidence) -> str:
        span = _fitted(clip, footage.duration_s)
        marks = ProofMarks(marks=evidence.marks)
        drawn = f"{footage.content_hash}|{span.start:.2f}|{span.end:.2f}|{marks.model_dump_json()}"
        key = f"ask/clips/{_digest(drawn)}.mp4"
        if await self._blobs.exists(key):
            return key
        source = await self._blobs.local_path(footage.original_key)
        with tempfile.TemporaryDirectory() as workdir:
            destination = Path(workdir) / "clip.mp4"
            if marks.marks:
                await self._maker.make_clip(source, destination, span, marks.marks)
            else:
                await self._video.cut_clip(source, destination, span)
            await self._blobs.write_file(key, destination)
        return key


# Long enough to see what happens, short enough for a chat; a long stretch shows its start.
def _fitted(span: TimeSpan, duration_s: float) -> TimeSpan:
    start, end = span.start, min(span.end, span.start + MAX_CLIP_S)
    if end - start < MIN_CLIP_S:
        middle = (start + end) / 2
        start, end = middle - MIN_CLIP_S / 2, middle + MIN_CLIP_S / 2
    return TimeSpan(start=max(0.0, start), end=min(duration_s, end))


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:24]
