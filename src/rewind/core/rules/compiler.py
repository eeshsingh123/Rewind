import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from rewind.contracts import Charge, Checkpoint, Predicate, RuleSpec
from rewind.core.ports import (
    AssistantMessage,
    BlobStore,
    ChatMessage,
    CompletionRequest,
    LanguageModel,
    LanguageModelError,
    SystemMessage,
    UserMessage,
)
from rewind.core.rules.engine import zone_keys
from rewind.core.rules.predicates import PREDICATES

MAX_ATTEMPTS = 3
MAX_OUTPUT_TOKENS = 2048
CACHE_PREFIX = "rules/compiled"


class RuleCompileError(Exception):
    def __init__(self, message: str, charges: tuple[Charge, ...]) -> None:
        super().__init__(message)
        # Calls made before the failure are still billed, so the caller can still record them.
        self.charges = charges


@dataclass(frozen=True, slots=True)
class CompiledRule:
    spec: RuleSpec
    # What each zone the rule names looks like, for finding it by text when it is not drawn.
    zone_phrases: Mapping[str, str]
    # Empty when the rule came from the cache.
    charges: tuple[Charge, ...]


class RuleCompiler:
    def __init__(self, model: LanguageModel, cache: BlobStore) -> None:
        self._model = model
        self._cache = cache

    async def compile(self, text: str) -> CompiledRule:
        normalized = normalize(text)
        key = self._cache_key(normalized)
        if await self._cache.exists(key):
            entry = _CacheEntry.model_validate_json(await self._cache.read(key))
            return CompiledRule(spec=entry.spec, zone_phrases=entry.zone_phrases, charges=())
        spec, zone_phrases, charges = await self._ask(normalized)
        entry = _CacheEntry(normalized_text=normalized, spec=spec, zone_phrases=zone_phrases)
        await self._cache.write(key, entry.model_dump_json().encode())
        return CompiledRule(spec=spec, zone_phrases=zone_phrases, charges=charges)

    async def _ask(self, sentence: str) -> tuple[RuleSpec, dict[str, str], tuple[Charge, ...]]:
        messages: list[ChatMessage] = [SystemMessage(SYSTEM_PROMPT), UserMessage(sentence)]
        charges: list[Charge] = []
        error = ""
        for _ in range(MAX_ATTEMPTS):
            request = CompletionRequest(
                messages=tuple(messages),
                max_output_tokens=MAX_OUTPUT_TOKENS,
                response_schema=RESPONSE_SCHEMA,
                effort="minimal",
            )
            try:
                completion = await self._model.complete(request)
            except LanguageModelError as failure:
                raise RuleCompileError(str(failure), tuple(charges)) from failure
            charges.extend(completion.charges)
            try:
                spec, zone_phrases = _to_spec(completion.text)
            except ValueError as invalid:
                error = _describe(invalid)
                messages += [
                    AssistantMessage(text=completion.text),
                    UserMessage(
                        f"That JSON is not a valid rule: {error}. Reply with corrected JSON."
                    ),
                ]
                continue
            return spec, zone_phrases, tuple(charges)
        raise RuleCompileError(
            f"no valid rule after {MAX_ATTEMPTS} attempts: {error}", tuple(charges)
        )

    def _cache_key(self, normalized: str) -> str:
        version = re.sub(r"[^A-Za-z0-9._-]", "_", self._model.model_version)
        digest = hashlib.sha256(normalized.encode()).hexdigest()[:32]
        return f"{CACHE_PREFIX}/{version}/{digest}.json"


def normalize(text: str) -> str:
    collapsed = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return collapsed.rstrip(".!").rstrip()


class _CacheEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    normalized_text: str
    spec: RuleSpec
    zone_phrases: dict[str, str]


class _Reply(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _Named(_Reply):
    key: str
    phrase: str


class _Requirement(_Reply):
    kind: str
    part: str | None
    zone: str | None
    n: int | None
    timing: Literal["at_checkpoint", "ever_before"]


class _Order(_Reply):
    first: str
    then: str


class _Rule(_Reply):
    entity: _Named
    parts: list[_Named]
    zones: list[_Named]
    checkpoint: str
    checkpoint_zone: str | None
    requirements: list[_Requirement]
    order: list[_Order]
    dwell_zone: str | None
    dwell_max_s: float | None
    rate_min_per_minute: float | None
    unsupported: list[str]


_CHECKPOINT: TypeAdapter[Checkpoint] = TypeAdapter(Checkpoint)
_PREDICATE: TypeAdapter[Predicate] = TypeAdapter(Predicate)


def _to_spec(text: str) -> tuple[RuleSpec, dict[str, str]]:
    reply = _Rule.model_validate_json(text)
    if (reply.dwell_zone is None) != (reply.dwell_max_s is None):
        raise ValueError("dwell_zone and dwell_max_s must both be set or both be null")
    checkpoint = {"kind": reply.checkpoint, "zone": reply.checkpoint_zone}
    requirements = [
        {
            "predicate": _PREDICATE.validate_python(
                _without_nulls(r.model_dump(exclude={"timing"}))
            ),
            "timing": r.timing,
        }
        for r in reply.requirements
    ]
    spec = RuleSpec.model_validate(
        {
            "entity": reply.entity.model_dump(),
            "parts": {part.key: part.phrase for part in reply.parts},
            "checkpoint": _CHECKPOINT.validate_python(_without_nulls(checkpoint)),
            "requirements": requirements,
            "order": [step.model_dump() for step in reply.order],
            "dwell": None
            if reply.dwell_zone is None
            else {"zone": reply.dwell_zone, "max_s": reply.dwell_max_s},
            "rate": None
            if reply.rate_min_per_minute is None
            else {"min_per_minute": reply.rate_min_per_minute},
            "unsupported": reply.unsupported,
        }
    )
    phrases = {zone.key: zone.phrase for zone in reply.zones}
    unnamed = sorted(zone_keys(spec) - phrases.keys())
    if unnamed:
        raise ValueError(f"zones used but not listed in zones: {', '.join(unnamed)}")
    return spec, {key: phrases[key] for key in sorted(zone_keys(spec))}


def _without_nulls(fields: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in fields.items() if value is not None}


def _describe(error: ValueError) -> str:
    if not isinstance(error, ValidationError):
        return str(error)
    return "; ".join(
        f"{'.'.join(str(part) for part in issue['loc']) or 'reply'}: {issue['msg']}"
        for issue in error.errors(include_url=False)
    )


_NAMED: dict[str, JsonValue] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["key", "phrase"],
    "properties": {"key": {"type": "string"}, "phrase": {"type": "string"}},
}

_NULLABLE_STRING: dict[str, JsonValue] = {"type": ["string", "null"]}

RESPONSE_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "entity",
        "parts",
        "zones",
        "checkpoint",
        "checkpoint_zone",
        "requirements",
        "order",
        "dwell_zone",
        "dwell_max_s",
        "rate_min_per_minute",
        "unsupported",
    ],
    "properties": {
        "entity": _NAMED,
        "parts": {"type": "array", "items": _NAMED},
        "zones": {"type": "array", "items": _NAMED},
        "checkpoint": {"type": "string", "enum": ["enters", "exits_view", "end_of_footage"]},
        "checkpoint_zone": _NULLABLE_STRING,
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["kind", "part", "zone", "n", "timing"],
                "properties": {
                    "kind": {"type": "string", "enum": list[JsonValue](sorted(PREDICATES))},
                    "part": _NULLABLE_STRING,
                    "zone": _NULLABLE_STRING,
                    "n": {"type": ["integer", "null"]},
                    "timing": {"type": "string", "enum": ["at_checkpoint", "ever_before"]},
                },
            },
        },
        "order": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["first", "then"],
                "properties": {"first": {"type": "string"}, "then": {"type": "string"}},
            },
        },
        "dwell_zone": _NULLABLE_STRING,
        "dwell_max_s": {"type": ["number", "null"]},
        "rate_min_per_minute": {"type": ["number", "null"]},
        "unsupported": {"type": "array", "items": {"type": "string"}},
    },
}


def _example(sentence: str, rule: dict[str, object]) -> str:
    defaults: dict[str, object] = {
        "parts": [],
        "zones": [],
        "checkpoint": "exits_view",
        "checkpoint_zone": None,
        "requirements": [],
        "order": [],
        "dwell_zone": None,
        "dwell_max_s": None,
        "rate_min_per_minute": None,
        "unsupported": [],
    }
    return f"Sentence: {sentence}\nJSON: {json.dumps({'entity': rule['entity']} | defaults | rule)}"


def _need(kind: str, timing: str = "at_checkpoint", **fields: object) -> dict[str, object]:
    return {"kind": kind, "part": None, "zone": None, "n": None, "timing": timing} | fields


_EXAMPLES = "\n\n".join(
    (
        _example(
            "Every bottle needs a lid and a sticker before it leaves.",
            {
                "entity": {"key": "bottle", "phrase": "bottle"},
                "parts": [{"key": "lid", "phrase": "lid"}, {"key": "sticker", "phrase": "sticker"}],
                "requirements": [_need("has", part="lid"), _need("has", part="sticker")],
            },
        ),
        _example(
            "Each pallet must have a wrap when it reaches the dock and must weigh under 500 kg.",
            {
                "entity": {"key": "pallet", "phrase": "pallet"},
                "parts": [{"key": "wrap", "phrase": "plastic stretch wrap"}],
                "zones": [{"key": "dock", "phrase": "loading dock"}],
                "checkpoint": "enters",
                "checkpoint_zone": "dock",
                "requirements": [_need("has", part="wrap")],
                "unsupported": ["must weigh under 500 kg"],
            },
        ),
        _example(
            "Each tray must hold 6 cans before the film wrap.",
            {
                "entity": {"key": "tray", "phrase": "tray"},
                "parts": [
                    {"key": "can", "phrase": "can"},
                    {"key": "film_wrap", "phrase": "plastic film wrap"},
                ],
                "requirements": [_need("count_at_least", "ever_before", part="can", n=6)],
                "order": [{"first": "can", "then": "film_wrap"}],
            },
        ),
        _example(
            "No box may sit in the scale area for longer than 2 minutes.",
            {
                "entity": {"key": "box", "phrase": "box"},
                "zones": [{"key": "scale", "phrase": "weighing scale"}],
                "dwell_zone": "scale",
                "dwell_max_s": 120,
            },
        ),
        _example(
            "The line must ship at least 600 bags per hour.",
            {"entity": {"key": "bag", "phrase": "bag"}, "rate_min_per_minute": 10},
        ),
        _example(
            "Every crate must pass through the wash zone and have a barcode by the end of the "
            "video.",
            {
                "entity": {"key": "crate", "phrase": "crate"},
                "parts": [{"key": "barcode", "phrase": "barcode"}],
                "zones": [{"key": "wash", "phrase": "washing station"}],
                "checkpoint": "end_of_footage",
                "requirements": [
                    _need("in_zone", "ever_before", zone="wash"),
                    _need("has", part="barcode"),
                ],
            },
        ),
    )
)

SYSTEM_PROMPT = f"""\
You translate one plain-English rule for a production or packing line into JSON. A camera \
watches the line. A segmentation model finds each item (the entity) and its parts from short \
noun phrases. A deterministic engine then checks the rule for every item.

The rule language:
- entity: the kind of item the rule is about. key is snake_case; phrase is a short noun phrase.
- parts: everything the item must have, one entry per part. key is snake_case of the part's \
noun ("shipping label" -> shipping_label, "QR code" -> qr_code); phrase is a short singular \
noun phrase describing how the part looks, for the segmentation model. A state the camera \
can see belongs in the phrase ("4 flaps down" -> phrase "folded-down flap"), not in unsupported.
- zones: every named area the rule uses. key is snake_case of the name the user gives, without \
generic words such as zone, area or station ("the capper zone" -> capper); phrase is a short \
noun phrase describing how the area looks.
- checkpoint: the moment each item is checked.
  - "exits_view": when it leaves the view, leaves, exits, or passes. Use this when the sentence \
names no moment.
  - "enters" with checkpoint_zone: when it enters or reaches a named zone.
  - "end_of_footage": at the end of the footage or video.
  checkpoint_zone is null unless checkpoint is "enters".
- requirements: a list with one entry per thing the item must satisfy. Never merge two needs \
into one entry, and never drop one.
  - "has" with part: the part is present.
  - "count_at_least" with part and n: at least n of the part are visible at once.
  - "in_zone" with zone: the item is inside the zone.
  Set the fields a kind does not use to null.
  timing "at_checkpoint": it must hold at the checkpoint ("needs a cap before it leaves" or \
"when it enters" both mean at the checkpoint). timing "ever_before": it must have held at some \
point before the checkpoint ("must go through", "at some point", or a need before another part).
- order: a list of {{first, then}} part keys, when one part must appear before another.
- dwell_zone and dwell_max_s: an item must not stay in the zone longer than dwell_max_s seconds.
- rate_min_per_minute: at least this many items per minute must reach the checkpoint.
- unsupported: every part of the sentence the language cannot express (colour, size, weight, \
text content, people, anything else), quoted from the sentence. Express everything else.
Durations are in seconds and rates per minute: convert other units. Use only parts and zones \
the sentence names. Every key in requirements and order must be listed in parts or zones.

Examples:

{_EXAMPLES}
"""
