import hashlib
import json
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from rewind.contracts import Charge, Footage, TimeSpan
from rewind.core.perception import ProxyMaker
from rewind.core.ports import (
    BlobStore,
    CompletionRequest,
    LanguageModel,
    LanguageModelError,
    SystemMessage,
    UserMessage,
)

# Muse samples the video itself; this only bounds the size of the clip it is sent.
WATCH_FPS = 10.0
MAX_OUTPUT_TOKENS = 6000
_PROMPT_VERSION = "1"

WATCHING = (
    "You watch video footage for someone in charge of the place it shows: a line manager, "
    "a shop owner, a site lead. Write plain, short sentences. Give every time in seconds from "
    "the start of the clip you are sent. Say only what you can see or hear; when something "
    "cannot be told from the video, say so."
)
FIRST_LOOK = (
    "Watch all of this {length:.1f} s video.\n"
    "summary: two plain sentences on what it shows.\n"
    "moments: every notable moment (something stopping or starting, an item missing a part "
    "or looking wrong, anything unusual), each with its start and end in seconds.\n"
    "questions: 3 to 5 short questions the person in charge would ask about this video and "
    "that the video can answer."
)
WATCH = (
    "This clip is {length:.1f} s long. Question: {question}\n"
    "answer: a plain answer from this clip only.\n"
    "moments: the moments that back the answer, each with its start and end in seconds."
)

_MOMENTS: dict[str, JsonValue] = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["start_s", "end_s", "what"],
        "properties": {
            "start_s": {"type": "number"},
            "end_s": {"type": "number"},
            "what": {"type": "string"},
        },
    },
}
FIRST_LOOK_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "moments", "questions"],
    "properties": {
        "summary": {"type": "string"},
        "moments": _MOMENTS,
        "questions": {"type": "array", "items": {"type": "string"}},
    },
}
WATCH_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer", "moments"],
    "properties": {"answer": {"type": "string"}, "moments": _MOMENTS},
}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Moment(_Frozen):
    # Footage time, already shifted from the clip Muse watched.
    span: TimeSpan
    what: str


class FirstLook(_Frozen):
    summary: str
    moments: tuple[Moment, ...]
    questions: tuple[str, ...]


class Watched(_Frozen):
    answer: str
    moments: tuple[Moment, ...]


class _RawMoment(BaseModel):
    start_s: float
    end_s: float
    what: str


class _RawFirstLook(BaseModel):
    summary: str
    moments: list[_RawMoment]
    questions: list[str]


class _RawWatched(BaseModel):
    answer: str
    moments: list[_RawMoment]


# Muse on the whole video or one stretch of it. Each answer is kept by its inputs, so the same
# look never costs twice.
class Watcher:
    def __init__(self, blobs: BlobStore, proxies: ProxyMaker) -> None:
        self._blobs = blobs
        self._proxies = proxies

    async def first_look(
        self, footage: Footage, model: LanguageModel
    ) -> tuple[FirstLook, tuple[Charge, ...]]:
        span = TimeSpan(start=0, end=footage.duration_s)
        prompt = FIRST_LOOK.format(length=footage.duration_s)
        raw, charges = await self._ask(
            footage, span, prompt, FIRST_LOOK_SCHEMA, _RawFirstLook, model
        )
        look = FirstLook(
            summary=raw.summary.strip(),
            moments=_moments(raw.moments, span),
            questions=tuple(q.strip() for q in raw.questions if q.strip())[:5],
        )
        return look, charges

    async def watch(
        self, footage: Footage, span: TimeSpan, question: str, model: LanguageModel
    ) -> tuple[Watched, tuple[Charge, ...]]:
        prompt = WATCH.format(length=span.end - span.start, question=question)
        raw, charges = await self._ask(footage, span, prompt, WATCH_SCHEMA, _RawWatched, model)
        return Watched(answer=raw.answer.strip(), moments=_moments(raw.moments, span)), charges

    async def _ask[T: BaseModel](
        self,
        footage: Footage,
        span: TimeSpan,
        prompt: str,
        schema: Mapping[str, JsonValue],
        reply: type[T],
        model: LanguageModel,
    ) -> tuple[T, tuple[Charge, ...]]:
        inputs = [footage.content_hash, span.start, span.end, prompt, model.model_version,
                  _PROMPT_VERSION]  # fmt: skip
        key = f"ask/watched/{hashlib.sha256(json.dumps(inputs).encode()).hexdigest()[:24]}.txt"
        if await self._blobs.exists(key):
            return _parse(reply, (await self._blobs.read(key)).decode()), ()
        clip = await self._proxies.ensure(
            footage.content_hash, footage.original_key, [span], WATCH_FPS
        )
        completion = await model.complete(
            CompletionRequest(
                messages=(
                    SystemMessage(WATCHING),
                    UserMessage(prompt, video_key=clip.proxy.blob_key),
                ),
                max_output_tokens=MAX_OUTPUT_TOKENS,
                response_schema=schema,
                effort="low",
            )
        )
        parsed = _parse(reply, completion.text)
        await self._blobs.write(key, completion.text.encode())
        return parsed, completion.charges


def _parse[T: BaseModel](model: type[T], text: str) -> T:
    try:
        return model.model_validate_json(text)
    except ValidationError as error:
        raise LanguageModelError(f"Muse's look did not fit its schema: {error}") from error


def _moments(raw: list[_RawMoment], span: TimeSpan) -> tuple[Moment, ...]:
    found: list[Moment] = []
    for moment in raw:
        start = min(max(span.start + moment.start_s, span.start), span.end)
        end = min(max(span.start + moment.end_s, start), span.end)
        if moment.what.strip():
            found.append(Moment(span=TimeSpan(start=start, end=end), what=moment.what.strip()))
    return tuple(sorted(found, key=lambda m: m.span.start))
