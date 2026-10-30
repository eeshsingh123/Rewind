import hashlib
import tempfile
from pathlib import Path

from rewind.contracts import Footage, TimeSpan
from rewind.core.analysis import ClipMaker, ProofMarks
from rewind.core.analysis.facts import MAX_CLIP_S
from rewind.core.ask.evidence import Evidence
from rewind.core.ports import BlobStore, VideoTools

MIN_CLIP_S = 3.0


# The proof behind a claim: its evidence's stretch, with the boxes drawn, made once.
class ProofClips:
    def __init__(self, blobs: BlobStore, video: VideoTools, clips: ClipMaker) -> None:
        self._blobs = blobs
        self._video = video
        self._clips = clips

    async def render(self, footage: Footage, evidence: Evidence) -> str | None:
        if evidence.clip is None:
            return None
        span = _fitted(evidence.clip, footage.duration_s)
        marks = ProofMarks(marks=evidence.marks)
        drawn = f"{footage.content_hash}|{span.start:.2f}|{span.end:.2f}|{marks.model_dump_json()}"
        key = f"ask/clips/{hashlib.sha256(drawn.encode()).hexdigest()[:24]}.mp4"
        if await self._blobs.exists(key):
            return key
        source = await self._blobs.local_path(footage.original_key)
        with tempfile.TemporaryDirectory() as workdir:
            destination = Path(workdir) / "clip.mp4"
            if marks.marks:
                await self._clips.make_clip(source, destination, span, marks.marks)
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
