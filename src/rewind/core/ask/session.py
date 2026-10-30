from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ConfigDict

from rewind.contracts import Charge, Footage, TimeSpan
from rewind.core.ask.watch import FirstLook, Watcher
from rewind.core.perception import Ingest
from rewind.core.ports import BlobStore, LanguageModel

# Told what is happening, in plain words, as it happens.
type OnStep = Callable[["Step"], Awaitable[None]]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Step(_Frozen):
    text: str
    done: bool


class Exchange(_Frozen):
    question: str
    answer: str


# One video and everything asked about it. There are no lines and no rules to set up.
class Session(_Frozen):
    id: str
    footage: Footage
    # Where something moves; SAM is sent only these spans.
    active_spans: tuple[TimeSpan, ...]
    first_look: FirstLook
    exchanges: tuple[Exchange, ...] = ()


class Sessions:
    def __init__(self, blobs: BlobStore, ingest: Ingest, watcher: Watcher) -> None:
        self._blobs = blobs
        self._ingest = ingest
        self._watcher = watcher

    # Readies the video and has Muse watch it once: a summary, its moments, questions to try.
    async def open(
        self, session_id: str, original_key: str, model: LanguageModel, on_step: OnStep
    ) -> tuple[Session, tuple[Charge, ...]]:
        await on_step(Step(text="Getting the video ready", done=False))
        # A session is its own group of footage: there are no lines any more.
        ingested = await self._ingest.ingest(session_id, session_id, original_key)
        await on_step(Step(text="Watching the whole video", done=False))
        look, charges = await self._watcher.first_look(ingested.footage, model)
        await on_step(Step(text="Watched the whole video", done=True))
        session = Session(
            id=session_id,
            footage=ingested.footage,
            active_spans=ingested.active_spans,
            first_look=look,
        )
        await self.save(session)
        return session, charges

    async def get(self, session_id: str) -> Session | None:
        key = _key(session_id)
        if not await self._blobs.exists(key):
            return None
        return Session.model_validate_json(await self._blobs.read(key))

    async def save(self, session: Session) -> None:
        await self._blobs.write(_key(session.id), session.model_dump_json().encode())


def _key(session_id: str) -> str:
    return f"ask/sessions/{session_id}.json"
