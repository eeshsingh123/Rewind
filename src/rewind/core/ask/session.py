import hashlib
import tempfile
from collections.abc import Awaitable, Callable
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from rewind.contracts import Box, Charge, Footage, TimeSpan
from rewind.core.analysis import still_key
from rewind.core.ask.evidence import Kind
from rewind.core.ask.watch import FirstLook, Watcher
from rewind.core.perception import Ingest
from rewind.core.ports import BlobStore, LanguageModel, VideoTools

# Told what is happening, in plain words, as it happens.
type OnStep = Callable[["Step"], Awaitable[None]]

_PHOTO_TYPES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".avif", ".bmp"})


def is_photo(name: str) -> bool:
    return PurePosixPath(name).suffix.lower() in _PHOTO_TYPES


class PhotoReader(Protocol):
    async def upright(self, source: Path, destination: Path) -> tuple[int, int]:
        """Writes the photo as an upright JPEG; returns its width and height."""
        ...


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Step(_Frozen):
    text: str
    done: bool


# The box a person dragged on a frame; a photo's one frame is at time 0.
class Marked(_Frozen):
    box: Box
    t: float


class Finding(_Frozen):
    text: str
    kind: Kind
    # The proof's blob key, a clip or on a photo an image; None when there is nothing to show.
    proof_key: str | None
    # Where it happened, for a timeline; None on a photo or when no one moment backs it.
    span: TimeSpan | None = None


class Exchange(_Frozen):
    question: str
    # All the findings as one text; what the agent reads back in a later question.
    answer: str
    findings: tuple[Finding, ...] = ()
    # Each finished step, so a page opened later shows how the answer was reached.
    steps: tuple[str, ...] = ()
    cost_usd: Decimal = Decimal(0)
    marked: Marked | None = None


# One photo or video and everything asked about it. There are no lines and no rules to set up.
class Session(_Frozen):
    id: str
    # A photo is footage of one frame, at time 0 of a one-second clip.
    footage: Footage
    # Where something moves; SAM is sent only these spans. A photo has none.
    active_spans: tuple[TimeSpan, ...]
    first_look: FirstLook
    exchanges: tuple[Exchange, ...] = ()
    photo: bool = False
    # What a browser can play: a photo's JPEG, or a video made playable whatever it was shot in.
    playback_key: str = ""


class Sessions:
    def __init__(
        self,
        blobs: BlobStore,
        ingest: Ingest,
        watcher: Watcher,
        photos: PhotoReader,
        video: VideoTools,
    ) -> None:
        self._blobs = blobs
        self._ingest = ingest
        self._watcher = watcher
        self._photos = photos
        self._video = video

    # Readies the upload and has Muse go over it once: a summary, what stands out, questions to try.
    async def open(
        self, session_id: str, original_key: str, model: LanguageModel, on_step: OnStep
    ) -> tuple[Session, tuple[Charge, ...]]:
        if is_photo(original_key):
            return await self._open_photo(session_id, original_key, model, on_step)
        await on_step(Step(text="Getting the video ready", done=False))
        # A session is its own group of footage: there are no lines any more.
        ingested = await self._ingest.ingest(session_id, session_id, original_key)
        playback_key = await self._playback(ingested.footage)
        await on_step(Step(text="Watching the whole video", done=False))
        look, charges = await self._watcher.first_look(ingested.footage, model)
        await on_step(Step(text="Watched the whole video", done=True))
        session = Session(
            id=session_id,
            footage=ingested.footage,
            active_spans=ingested.active_spans,
            first_look=look,
            playback_key=playback_key,
        )
        await self.save(session)
        return session, charges

    # Made once per video content; phones shoot formats that most browsers cannot play.
    async def _playback(self, footage: Footage) -> str:
        key = f"ask/playback/{footage.content_hash}.mp4"
        if await self._blobs.exists(key):
            return key
        source = await self._blobs.local_path(footage.original_key)
        with tempfile.TemporaryDirectory() as workdir:
            destination = Path(workdir) / "playback.mp4"
            await self._video.playable(source, destination)
            await self._blobs.write_file(key, destination)
        return key

    async def _open_photo(
        self, session_id: str, original_key: str, model: LanguageModel, on_step: OnStep
    ) -> tuple[Session, tuple[Charge, ...]]:
        await on_step(Step(text="Looking at the photo", done=False))
        footage = await self._photo_footage(session_id, original_key)
        look, charges = await self._watcher.first_look_at_photo(footage, model)
        await on_step(Step(text="Looked at the photo", done=True))
        session = Session(
            id=session_id,
            footage=footage,
            active_spans=(),
            first_look=look,
            photo=True,
            playback_key=footage.original_key,
        )
        await self.save(session)
        return session, charges

    async def _photo_footage(self, session_id: str, original_key: str) -> Footage:
        content_hash = hashlib.sha256(await self._blobs.read(original_key)).hexdigest()
        source = await self._blobs.local_path(original_key)
        key = still_key(session_id, 0)
        with tempfile.TemporaryDirectory() as workdir:
            destination = Path(workdir) / "photo.jpg"
            width, height = await self._photos.upright(source, destination)
            await self._blobs.write_file(key, destination)
        return Footage(
            id=session_id,
            line_id=session_id,
            content_hash=content_hash,
            original_key=key,
            duration_s=1.0,
            fps=1.0,
            width=width,
            height=height,
            status="ready",
        )

    async def get(self, session_id: str) -> Session | None:
        key = _key(session_id)
        if not await self._blobs.exists(key):
            return None
        return Session.model_validate_json(await self._blobs.read(key))

    async def save(self, session: Session) -> None:
        await self._blobs.write(_key(session.id), session.model_dump_json().encode())

    async def remembered(self, digest: str) -> Exchange | None:
        key = _answer_key(digest)
        if not await self._blobs.exists(key):
            return None
        return Exchange.model_validate_json(await self._blobs.read(key))

    async def remember(self, digest: str, exchange: Exchange) -> None:
        await self._blobs.write(_answer_key(digest), exchange.model_dump_json().encode())


def _key(session_id: str) -> str:
    return f"ask/sessions/{session_id}.json"


def _answer_key(digest: str) -> str:
    return f"ask/answers/{digest}.json"
