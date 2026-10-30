import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from rewind.contracts import Charge
from rewind.core.ask.session import Marked, Session
from rewind.core.perception import PartLooker
from rewind.core.ports import (
    BlobStore,
    CompletionRequest,
    LanguageModel,
    LanguageModelError,
    SystemMessage,
    UserMessage,
)

_MAX_OUTPUT_TOKENS = 3000
_PROMPT_VERSION = "2"

READING = """\
You read one question about a {media} before anyone works on it, so that nothing gets assumed. \
You get what a first look found, one frame, and what was asked before. Use them to work out what \
the person's words mean in this footage.

thing: the kind of thing asked about, as a plain noun without any place or condition ("cardboard \
box"). Empty when the question is not about separate items.
conditions: each thing to check about one item, as its own yes or no question, in English. Add \
none that were not asked. Empty when there is nothing to check item by item.
combine: "separate" when several conditions are listed, numbered or not: each gets its own \
answer. "all" only when the person says in words that an item must meet them together, such as \
"both" or "and also". "any" only when the person says either will do. "none" for fewer than two \
conditions.
part: the part of the frame the question is limited to, in plain words. Empty when it covers all \
of it. Direction words belong to whatever they describe: in "flaps open to the right" the right \
is how a flap opens, not a part of the frame.
plain: one short sentence, in the language of the question, saying how you read it, in the \
person's own words, naming what you will check and how the answers are given.
unclear: only when the words can mean two things that give different answers and the footage does \
not settle which: one short question to ask back, in the language of the question. Otherwise empty.
choices: when unclear is set, 2 or 3 short answers to it, each making sense on its own. Otherwise \
empty."""
READ_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["thing", "conditions", "combine", "part", "plain", "unclear", "choices"],
    "properties": {
        "thing": {"type": "string"},
        "conditions": {"type": "array", "items": {"type": "string"}},
        "combine": {"type": "string", "enum": ["separate", "all", "any", "none"]},
        "part": {"type": "string"},
        "plain": {"type": "string"},
        "unclear": {"type": "string"},
        "choices": {"type": "array", "items": {"type": "string"}},
    },
}


class Reading(BaseModel):
    model_config = ConfigDict(frozen=True)
    thing: str
    conditions: tuple[str, ...]
    combine: Literal["separate", "all", "any", "none"]
    part: str
    plain: str
    unclear: str
    choices: tuple[str, ...]

    # What the agent is told, so it works from the reading and not from a guess.
    @property
    def brief(self) -> str:
        lines = ["How the question was read:"]
        if self.thing:
            lines.append(f"Thing: {self.thing}")
        lines += [f"Check: {condition}" for condition in self.conditions]
        if len(self.conditions) > 1:
            lines.append(f"Combine: {self.combine}")
        if self.part:
            lines.append(f"Part of the frame: {self.part}")
        return "\n".join(lines)


# One call before any work, kept by its inputs, so a question is read once.
class Reader:
    def __init__(self, blobs: BlobStore, stills: PartLooker) -> None:
        self._blobs = blobs
        self._stills = stills

    async def read(
        self,
        session: Session,
        question: str,
        marked: Marked | None,
        model: LanguageModel,
        charges: list[Charge],
    ) -> Reading:
        footage = session.footage
        told = _told(session, marked)
        inputs = [footage.content_hash, told, question, model.model_version, _PROMPT_VERSION]
        key = f"ask/readings/{hashlib.sha256(json.dumps(inputs).encode()).hexdigest()[:24]}.json"
        if await self._blobs.exists(key):
            return _parsed((await self._blobs.read(key)).decode())
        media = "photo" if session.photo else "video"
        image_key = (
            footage.original_key
            if session.photo
            else await self._stills.still(footage, footage.duration_s / 2)
        )
        completion = await model.complete(
            CompletionRequest(
                messages=(
                    SystemMessage(READING.format(media=media)),
                    UserMessage(f"{told}\n\nQuestion: {question}", image_keys=(image_key,)),
                ),
                max_output_tokens=_MAX_OUTPUT_TOKENS,
                response_schema=READ_SCHEMA,
                effort="low",
            )
        )
        charges += completion.charges
        reading = _parsed(completion.text)
        await self._blobs.write(key, completion.text.encode())
        return reading


def _told(session: Session, marked: Marked | None) -> str:
    look = session.first_look
    lines = [f"First look: {look.summary}", *(f"- {moment.what}" for moment in look.moments)]
    lines += [f"Asked before: {e.question}\nAnswered: {e.answer}" for e in session.exchanges]
    if marked is not None:
        lines.append("The person drew a box on the frame: that is the part they mean.")
    return "\n".join(lines)


def _parsed(text: str) -> Reading:
    try:
        return Reading.model_validate_json(text)
    except ValidationError as error:
        raise LanguageModelError(f"Muse's reading did not fit its schema: {error}") from error
