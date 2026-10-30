import hashlib
import json
from dataclasses import dataclass

from pydantic import BaseModel, JsonValue, ValidationError

from rewind.contracts import Box, Charge, Footage
from rewind.core.analysis import Mark, PartLooker, Sighting
from rewind.core.perception import frame_index
from rewind.core.ports import (
    BlobStore,
    CompletionRequest,
    LanguageModel,
    LanguageModelError,
    Segmenter,
    UserMessage,
)

# Muse's coordinates run 0 to 1000 over the image it was sent.
_GRID = 1000
_MAX_OUTPUT_TOKENS = 1000
_PROMPT_VERSION = "1"

MARK = (
    "You are an object grounding expert. Provide the bounding box coordinates of this part of "
    'the image: "{name}". Cover all of it and nothing else. Answer with "areas": one entry in '
    'the format {{"object_name": "<name>", "bbox": {{"x_min": <int>, "y_min": <int>, "x_max": '
    '<int>, "y_max": <int>}}, "holds": "<noun>"}}. The coordinates should be in the 0-1000 '
    'range, with (0,0) at the top-left. "holds" is the one kind of object that most fills that '
    'part, as a short noun phrase. If you cannot find that part, return an empty "areas".'
)
MARK_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["areas"],
    "properties": {
        "areas": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["object_name", "bbox", "holds"],
                "properties": {
                    "object_name": {"type": "string"},
                    "holds": {"type": "string"},
                    "bbox": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["x_min", "y_min", "x_max", "y_max"],
                        "properties": {
                            "x_min": {"type": "number"},
                            "y_min": {"type": "number"},
                            "x_max": {"type": "number"},
                            "y_max": {"type": "number"},
                        },
                    },
                },
            },
        }
    },
}


@dataclass(frozen=True, slots=True)
class Area:
    # As the agent named it.
    name: str
    box: Box
    # The frame it was marked on.
    t: float

    def holds(self, box: Box) -> bool:
        return _centre_inside(box, self.box)

    @property
    def mark(self) -> Mark:
        return Mark(kind="area", tag=short_tag(self.name), path=(Sighting(t=self.t, box=self.box),))


def short_tag(text: str) -> str:
    return text if len(text) <= 24 else text[:23] + "…"


class _Bbox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class _Marked(BaseModel):
    bbox: _Bbox
    holds: str


class _Marks(BaseModel):
    areas: list[_Marked]


# Muse marks the area and names what fills it; SAM must then find that thing at the mark. A box
# prompt could not catch a wrong mark: SAM outlines whatever is under a box, while a concept
# prompt finds only that thing.
class AreaMarker:
    def __init__(self, blobs: BlobStore, stills: PartLooker) -> None:
        self._blobs = blobs
        self._stills = stills

    async def mark(
        self,
        footage: Footage,
        name: str,
        t: float,
        model: LanguageModel,
        segmenter: Segmenter,
        charges: list[Charge],
    ) -> Area | None:
        marked = await self._ask_muse(footage, name, t, model, charges)
        box = _box(marked.bbox) if marked else None
        if marked is None or box is None or not marked.holds.strip():
            return None
        thing = " ".join(marked.holds.lower().split())
        looks = await self._stills.look(footage, {thing: {t}}, segmenter)
        charges += looks.charges
        if any(_agree(box, seen.box) for seen in looks.observations):
            return Area(name, box, t)
        return None

    # Kept by its inputs, so the same area on the same frame is never paid for twice.
    async def _ask_muse(
        self, footage: Footage, name: str, t: float, model: LanguageModel, charges: list[Charge]
    ) -> _Marked | None:
        frame = frame_index(t, footage.fps)
        inputs = [footage.content_hash, frame, name, model.model_version, _PROMPT_VERSION]
        key = f"ask/areas/{hashlib.sha256(json.dumps(inputs).encode()).hexdigest()[:24]}.json"
        if await self._blobs.exists(key):
            return _first((await self._blobs.read(key)).decode())
        image_key = await self._stills.still(footage, t)
        completion = await model.complete(
            CompletionRequest(
                # Muse grounds best from a direct request next to the image, with no system prompt.
                messages=(UserMessage(MARK.format(name=name), image_keys=(image_key,)),),
                max_output_tokens=_MAX_OUTPUT_TOKENS,
                response_schema=MARK_SCHEMA,
                effort="low",
            )
        )
        charges += completion.charges
        marked = _first(completion.text)
        await self._blobs.write(key, completion.text.encode())
        return marked


def _first(text: str) -> _Marked | None:
    try:
        marks = _Marks.model_validate_json(text)
    except ValidationError as error:
        raise LanguageModelError(f"Muse's mark did not fit its schema: {error}") from error
    return marks.areas[0] if marks.areas else None


# Muse may step just past the grid at an edge.
def _box(bbox: _Bbox) -> Box | None:
    def unit(value: float) -> float:
        return min(1.0, max(0.0, value / _GRID))

    x0, y0, x1, y1 = unit(bbox.x_min), unit(bbox.y_min), unit(bbox.x_max), unit(bbox.y_max)
    return Box(x0=x0, y0=y0, x1=x1, y1=y1) if x0 < x1 and y0 < y1 else None


# They agree when either's middle lies in the other: the thing may be one item in the marked
# part, or larger than it.
def _agree(mark: Box, thing: Box) -> bool:
    return _centre_inside(thing, mark) or _centre_inside(mark, thing)


def _centre_inside(box: Box, outer: Box) -> bool:
    x, y = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
    return outer.x0 <= x <= outer.x1 and outer.y0 <= y <= outer.y1
