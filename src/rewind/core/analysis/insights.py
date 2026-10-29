import hashlib
import json
import logging
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from rewind.contracts import Box, Charge, Footage, Line, Rule, TimeSpan
from rewind.core.analysis.facts import (
    Earlier,
    Fact,
    RuleView,
    SheetInput,
    Tile,
    fact_sheet,
    picture_tiles,
    sheet_text,
    untraceable,
)
from rewind.core.analysis.proof import ProofMarks, marks_key
from rewind.core.analysis.replies import join_words, plural, rule_sentence
from rewind.core.analysis.timeline import Timeline
from rewind.core.ports import (
    AssistantMessage,
    BlobStore,
    ChatMessage,
    CompletionRequest,
    LanguageModel,
    LanguageModelError,
    Repository,
    SystemMessage,
    UserMessage,
    VideoToolsError,
)
from rewind.core.rules import at_border

logger = logging.getLogger(__name__)

# Bump when a prompt or the fact sheet changes, so cached answers are not reused.
PROMPT_VERSION = "3"
MAX_INSIGHTS = 3
MAX_TEXT_CHARS = 240
MAX_ANSWER_CHARS = 600
MAX_LABEL_CHARS = 24
MIN_WORDS = 3
MAX_QUESTION_CHARS = 500
PICTURE_TILES = 6
ANSWER_ATTEMPTS = 2
MAX_OUTPUT_TOKENS = 1024
# A quiet "nothing unusual" needs enough items to have looked at.
QUIET_FROM_ITEMS = 10
_FACT_ID = re.compile(r"\bF\d+\b")

FIND_PROMPT = """\
You are the analyst of Rewind, which checks camera footage of production and packing lines. \
Code has already found every item that broke a rule; the person sees those counts and times. \
Your job is to point out at most 3 things a line manager would act on, which the counts alone \
do not say.

You get numbered facts computed by code from the footage, and maybe one picture of tiles, \
each labelled with the fact id and time it shows (a red box is the item).

What is worth saying, best first:
- a stop and the output it cost;
- a fact marked notable: a pattern in when items failed, or a change from earlier recordings;
- a link between notable facts, such as failures right after a restart.
Never restate a rule's plain count on its own. Never say anything that is not notable.

Hard rules:
- Use only the facts. Every number and time you write must appear in a fact you cite, \
written the same way (times look like 00:14).
- Do not state a cause as certain. Say what a pattern points to only if its fact says so.
- The picture may help say what a failed item looks like; describe only what is plainly \
visible, and cite that item's fact.
- Plain words for a line manager: no fact ids, no statistics words like "p-value".
- Each insight: text (one sentence, at most 200 characters), facts (the ids it rests on), \
clip (one of those ids, marked "has a clip", whose clip proves it), label (a 2 to 4 word \
button name, such as "The stop").
- If nothing is worth saying, return an empty list. That is often the right answer.
"""

ANSWER_PROMPT = """\
You are the analyst of Rewind, which checks camera footage of production and packing lines. \
Answer the person's question about one recording, using only the numbered facts computed by \
code from it, and maybe one picture of tiles, each labelled with the fact id and time it shows.

Hard rules:
- Every number and time you write must appear in a fact you cite, written the same way \
(times look like 00:14). Never count or add up yourself: if the facts do not hold the number, \
the question cannot be answered.
- "When" and "how often" questions about failures are answered from the failure facts: \
the first one listed is when it started in this recording, and the rule's count is how \
often. Say that it is this recording only.
- The facts cover only what the line's rules check: the items, their parts, stops and pace. \
If the question needs anything else (why something happened, who did it, something no rule \
looks for, a time outside the recording, another recording), set answerable to false, say in \
one plain sentence what you cannot tell from the footage and why.
- suggest_rule: only when a new rule would let Rewind check it, one sentence in one of the \
forms Rewind can check, else an empty string. The forms: "every <item> has a <part> when it \
leaves the view" (or "when it enters the <area>", or "by the end of the video"), "every <item> \
gets its <part> before its <part>", "no <item> stays in the <area> longer than <n> seconds", \
"at least <n> <items> pass every minute". Rewind sees items and their parts, never people's \
identity, intent or anything off camera.
- Do not state a cause as certain; say what a pattern points to only if its fact says so.
- Plain, short words for a line manager: at most 3 sentences, no fact ids.
- clip: the id of the cited fact, marked "has a clip", that best shows the answer, or an \
empty string. label: a 2 to 4 word button name for that clip.
"""

_INSIGHT_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text", "facts", "clip", "label"],
    "properties": {
        "text": {"type": "string"},
        "facts": {"type": "array", "items": {"type": "string"}},
        "clip": {"type": "string"},
        "label": {"type": "string"},
    },
}
FIND_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["insights"],
    "properties": {"insights": {"type": "array", "items": _INSIGHT_SCHEMA}},
}
ANSWER_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answerable", "answer", "facts", "clip", "label", "suggest_rule"],
    "properties": {
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
        "facts": {"type": "array", "items": {"type": "string"}},
        "clip": {"type": "string"},
        "label": {"type": "string"},
        "suggest_rule": {"type": "string"},
    },
}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SavedInsight(_Frozen):
    id: str
    text: str
    label: str
    # The facts behind it, as code wrote them: what makes it provable.
    facts: tuple[str, ...]
    clip: TimeSpan | None
    # The failed items its clip draws.
    violation_ids: tuple[str, ...] = ()


class SavedInsights(_Frozen):
    found: tuple[SavedInsight, ...] = ()
    asked: tuple[SavedInsight, ...] = ()

    def get(self, insight_id: str) -> SavedInsight | None:
        return next((i for i in (*self.found, *self.asked) if i.id == insight_id), None)


class _Proposed(_Frozen):
    text: str
    facts: list[str]
    clip: str
    label: str


class _Found(_Frozen):
    insights: list[_Proposed]


class _Answered(_Frozen):
    answerable: bool
    answer: str
    facts: list[str]
    clip: str
    label: str
    suggest_rule: str


@dataclass(frozen=True, slots=True)
class Found:
    insights: tuple[SavedInsight, ...]
    # One line for a normal run; empty when there is nothing to say either way.
    quiet: str
    charges: tuple[Charge, ...]


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    insight: SavedInsight | None
    charges: tuple[Charge, ...]


class SheetMaker(Protocol):
    async def make_sheet(self, source: Path, destination: Path, tiles: Sequence[Tile]) -> None:
        """One JPEG of the footage at each tile's time, its box drawn and its label written."""
        ...


class InsightWriter:
    def __init__(self, model: LanguageModel, cache: BlobStore) -> None:
        self._model = model
        self._cache = cache

    async def find(
        self, facts: Sequence[Fact], picture_key: str | None
    ) -> tuple[tuple[SavedInsight, ...], tuple[Charge, ...]]:
        key = await self._cache_key("find", "", facts, picture_key)
        if await self._cache.exists(key):
            return _validated(_Found.model_validate_json(await self._cache.read(key)), facts), ()
        messages = _messages(FIND_PROMPT, sheet_text(facts), picture_key)
        try:
            completion = await self._model.complete(
                CompletionRequest(messages, MAX_OUTPUT_TOKENS, FIND_SCHEMA)
            )
        except LanguageModelError:
            logger.exception("the insight call failed")
            return (), ()
        try:
            found = _Found.model_validate_json(completion.text)
        except ValidationError:
            logger.warning("insights were not valid JSON: %s", completion.text[:200])
            return (), completion.charges
        await self._cache.write(key, found.model_dump_json().encode())
        return _validated(found, facts), completion.charges

    async def answer(self, question: str, facts: Sequence[Fact], picture_key: str | None) -> Answer:
        question = question.strip()[:MAX_QUESTION_CHARS]
        key = await self._cache_key("answer", question, facts, picture_key)
        if await self._cache.exists(key):
            answered = _Answered.model_validate_json(await self._cache.read(key))
            checked = _checked_answer(answered, facts)
            if not isinstance(checked, str):
                return Answer(*checked, ())
        sheet = f"{sheet_text(facts)}\n\nThe person asks: {question}"
        messages: list[ChatMessage] = list(_messages(ANSWER_PROMPT, sheet, picture_key))
        charges: list[Charge] = []
        for _ in range(ANSWER_ATTEMPTS):
            request = CompletionRequest(tuple(messages), MAX_OUTPUT_TOKENS, ANSWER_SCHEMA)
            try:
                completion = await self._model.complete(request)
            except LanguageModelError:
                logger.exception("the question call failed")
                break
            charges += completion.charges
            try:
                answered = _Answered.model_validate_json(completion.text)
            except ValidationError:
                problem = "That is not valid JSON for the schema."
            else:
                checked = _checked_answer(answered, facts)
                if not isinstance(checked, str):
                    await self._cache.write(key, answered.model_dump_json().encode())
                    return Answer(*checked, tuple(charges))
                problem = checked
            messages += [AssistantMessage(text=completion.text),
                         UserMessage(f"{problem} Reply with corrected JSON.")]  # fmt: skip
        return Answer(
            "I can't answer that reliably from what I found in this recording.",
            None,
            tuple(charges),
        )

    async def _cache_key(
        self, kind: str, question: str, facts: Sequence[Fact], picture_key: str | None
    ) -> str:
        picture = b""
        if picture_key is not None:
            picture = await self._cache.read(picture_key)
        payload = json.dumps(
            [PROMPT_VERSION, self._model.model_version, kind, question, sheet_text(facts),
             hashlib.sha256(picture).hexdigest()]
        )  # fmt: skip
        return f"insights/cache/{hashlib.sha256(payload.encode()).hexdigest()[:32]}.json"


class Insights:
    def __init__(
        self,
        repository: Repository,
        blobs: BlobStore,
        sheets: SheetMaker,
        writer_for: Callable[[str], InsightWriter],
    ) -> None:
        self._repository = repository
        self._blobs = blobs
        self._sheets = sheets
        self._writer_for = writer_for

    async def timeline(self, footage_id: str) -> Timeline | None:
        key = _timeline_key(footage_id)
        if not await self._blobs.exists(key):
            return None
        return Timeline.model_validate_json(await self._blobs.read(key))

    async def save_timeline(self, timeline: Timeline) -> None:
        await self._blobs.write(_timeline_key(timeline.footage_id),
                                timeline.model_dump_json().encode())  # fmt: skip

    async def saved(self, footage_id: str) -> SavedInsights:
        key = _saved_key(footage_id)
        if not await self._blobs.exists(key):
            return SavedInsights()
        return SavedInsights.model_validate_json(await self._blobs.read(key))

    # What stands out in one checked recording. A model call happens only when code found
    # something notable, and never twice for the same facts.
    async def find(
        self, workspace_id: str, line: Line, footage: Footage, numbers: Mapping[str, int]
    ) -> Found:
        given = await self._sheet_input(line, footage, numbers, None)
        if given is None:
            return Found((), "", ())
        facts = fact_sheet(given)
        insights: tuple[SavedInsight, ...] = ()
        charges: tuple[Charge, ...] = ()
        if any(fact.notable for fact in facts):
            picture = await self._picture(footage, facts)
            insights, charges = await self._writer_for(workspace_id).find(facts, picture)
        previous = await self.saved(footage.id)
        await self._save(footage.id, SavedInsights(found=insights, asked=previous.asked))
        notable = any(fact.notable for fact in facts)
        quiet = "" if insights or notable else _quiet(given)
        return Found(insights, quiet, charges)

    async def ask(
        self,
        workspace_id: str,
        line: Line,
        footage: Footage,
        numbers: Mapping[str, int],
        question: str,
        window: TimeSpan | None,
    ) -> Answer | None:
        """None when the recording has not been checked yet."""
        given = await self._sheet_input(line, footage, numbers, window)
        if given is None:
            return None
        if not given.timeline.judged:
            # Nothing to answer from, and a model would read "no facts" as "no such rule".
            entities = join_words(sorted({plural(view.entity, 2) for view in given.rules}))
            return Answer(
                f"I found no {entities or 'items'} the rules could judge in this recording, so "
                "I have nothing to answer from. If they are in view, try naming them the way "
                "they look on camera.",
                None,
                (),
            )
        facts = fact_sheet(given)
        picture = await self._picture(footage, facts)
        answer = await self._writer_for(workspace_id).answer(question, facts, picture)
        if answer.insight is not None and answer.insight.clip is not None:
            saved = await self.saved(footage.id)
            if saved.get(answer.insight.id) is None:
                asked = (*saved.asked, answer.insight)
                await self._save(footage.id, SavedInsights(found=saved.found, asked=asked))
        return answer

    async def _save(self, footage_id: str, saved: SavedInsights) -> None:
        await self._blobs.write(_saved_key(footage_id), saved.model_dump_json().encode())

    async def _sheet_input(
        self, line: Line, footage: Footage, numbers: Mapping[str, int], window: TimeSpan | None
    ) -> SheetInput | None:
        timeline = await self.timeline(footage.id)
        if timeline is None:
            return None
        rules = {rule.id: rule for rule in await self._repository.rules(line.id)}
        views = tuple(_view(rules[i], numbers.get(i, 0)) for i in timeline.rule_ids()
                      if i in rules and rules[i].active)  # fmt: skip
        checked = {view.rule_id for view in views}
        violations = tuple(v for v in await self._repository.violations(footage.id)
                           if v.rule_id in checked)  # fmt: skip
        return SheetInput(
            name=Path(footage.original_key).name,
            timeline=timeline,
            rules=views,
            violations=violations,
            earlier=await self._earlier(line, footage, checked),
            boxes={
                v.id: seen
                for v in violations
                if (seen := await self._last_whole(v.id, v.t)) is not None
            },
            window=window,
        )

    async def _earlier(
        self, line: Line, footage: Footage, rule_ids: set[str]
    ) -> dict[str, Earlier]:
        earlier: dict[str, Earlier] = {}
        for other in await self._repository.line_footage(line.id):
            if other.id == footage.id:
                continue
            timeline = await self.timeline(other.id)
            if timeline is None:
                continue
            for rule_id in rule_ids & set(timeline.rule_ids()):
                before = earlier.get(rule_id, Earlier(0, ()))
                earlier[rule_id] = Earlier(before.recordings + 1,
                                           before.judged + timeline.of_rule(rule_id))  # fmt: skip
        return earlier

    # Where the flagged item was last seen whole before it was judged, from its proof marks.
    async def _last_whole(self, violation_id: str, t: float) -> tuple[float, Box] | None:
        key = marks_key(violation_id)
        if not await self._blobs.exists(key):
            return None
        marks = ProofMarks.model_validate_json(await self._blobs.read(key))
        flagged = next((m for m in marks.marks if m.kind == "flagged"), None)
        if flagged is None or not flagged.path:
            return None
        whole = [s for s in flagged.path if s.t <= t and not at_border(s.box)]
        seen = whole[-1] if whole else min(flagged.path, key=lambda s: abs(s.t - t))
        return seen.t, seen.box

    async def _picture(self, footage: Footage, facts: Sequence[Fact]) -> str | None:
        tiles = picture_tiles(facts, PICTURE_TILES)
        if not tiles:
            return None
        drawn = hashlib.sha256(repr(tiles).encode()).hexdigest()[:12]
        key = f"insights/{footage.id}/picture-{drawn}.jpg"
        if await self._blobs.exists(key):
            return key
        source = await self._blobs.local_path(footage.original_key)
        try:
            with tempfile.TemporaryDirectory() as workdir:
                destination = Path(workdir) / "picture.jpg"
                await self._sheets.make_sheet(source, destination, tiles)
                await self._blobs.write_file(key, destination)
        except VideoToolsError:
            logger.exception("could not make the insight picture for %s", footage.id)
            return None
        return key


def _view(rule: Rule, number: int) -> RuleView:
    return RuleView(rule.id, number, rule_sentence(rule.spec), rule.spec.entity.phrase)


def _validated(found: _Found, facts: Sequence[Fact]) -> tuple[SavedInsight, ...]:
    kept: list[SavedInsight] = []
    for proposed in found.insights:
        checked = _checked(proposed.text, proposed.facts, proposed.clip, proposed.label, facts,
                           MAX_TEXT_CHARS, need_clip=True)  # fmt: skip
        if isinstance(checked, str):
            logger.info("dropped an insight (%s): %s", checked, proposed.text)
            continue
        if all(i.text != checked.text for i in kept):
            kept.append(checked)
    return tuple(kept[:MAX_INSIGHTS])


def _checked_answer(
    answered: _Answered, facts: Sequence[Fact]
) -> tuple[str, SavedInsight | None] | str:
    text = answered.answer.strip()
    if not answered.answerable:
        if len(text.split()) < MIN_WORDS:
            return "Say in a plain sentence what cannot be told from the footage."
        if untraceable(text, facts):
            return "Do not state numbers when the question cannot be answered."
        rule = answered.suggest_rule.strip()
        suggestion = f'\nYou could add a rule like: "{rule}"' if rule else ""
        return f"{text}{suggestion}", None
    checked = _checked(text, answered.facts, answered.clip, answered.label, facts,
                       MAX_ANSWER_CHARS, need_clip=False)  # fmt: skip
    if isinstance(checked, str):
        return checked
    return checked.text, checked if checked.clip is not None else None


# A sentence the facts it cites do not back is dropped: the reason, or the insight it makes.
def _checked(
    text: str,
    cited: Sequence[str],
    clip: str,
    label: str,
    facts: Sequence[Fact],
    max_chars: int,
    need_clip: bool,
) -> SavedInsight | str:
    by_id = {fact.id: fact for fact in facts}
    text = " ".join(text.split())
    if len(text.split()) < MIN_WORDS or len(text) > max_chars:
        return f"The text must be a plain sentence of at most {max_chars} characters."
    if _FACT_ID.search(text):
        return "Do not write fact ids in the text."
    unknown = [i for i in cited if i not in by_id]
    backing = [by_id[i] for i in cited if i in by_id]
    if unknown or not backing:
        return f"Cite only fact ids from the list (unknown: {', '.join(unknown) or 'none cited'})."
    missing = untraceable(text, backing)
    if missing:
        return (
            f"These numbers or times are not in the facts you cited: {', '.join(missing)}. "
            "Use only numbers written in the cited facts."
        )
    shown = by_id.get(clip) if clip in cited else None
    if shown is None or shown.clip is None:
        shown = next((fact for fact in backing if fact.clip is not None), None)
    if shown is None and need_clip:
        return "Cite a fact that has a clip."
    label = " ".join(label.split())
    if not label or len(label) > MAX_LABEL_CHARS:
        label = "Show the clip"
    digest = hashlib.sha256(f"{text}\x1f{shown.id if shown else ''}".encode()).hexdigest()[:10]
    return SavedInsight(
        id=digest,
        text=text,
        label=label,
        facts=tuple(fact.text for fact in backing),
        clip=shown.clip if shown else None,
        violation_ids=shown.violation_ids if shown else (),
    )


def _quiet(given: SheetInput) -> str:
    judged = {view.rule_id: len(given.timeline.of_rule(view.rule_id)) for view in given.rules}
    if max(judged.values(), default=0) < QUIET_FROM_ITEMS:
        return ""
    failed = len([v for v in given.violations if v.track_id is not None])
    entity = plural(given.rules[0].entity, 2) if len(given.rules) == 1 else "items"
    if failed >= 3:
        return (
            f"Nothing else stood out: the line didn't stop, and the {entity} that failed "
            "look spread out."
        )
    return "Nothing else stood out: the line didn't stop."


def _messages(prompt: str, sheet: str, picture_key: str | None) -> tuple[ChatMessage, ...]:
    images = (picture_key,) if picture_key is not None else ()
    return (SystemMessage(prompt), UserMessage(sheet, image_keys=images))


def _timeline_key(footage_id: str) -> str:
    return f"insights/{footage_id}/timeline.json"


def _saved_key(footage_id: str) -> str:
    return f"insights/{footage_id}/insights.json"
