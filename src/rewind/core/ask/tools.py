import hashlib
import json
import tempfile
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from rewind.contracts import (
    Charge,
    Footage,
    Observation,
    SamplingPlan,
    SamplingWindow,
    TimeSpan,
    Track,
)
from rewind.core.analysis import SheetMaker, Sighting, Tile, clock_time, join_words, naive_tracks
from rewind.core.analysis.proof import Mark
from rewind.core.analysis.replies import plural
from rewind.core.analysis.timeline import Judged, find_stops, usual_gap
from rewind.core.ask.evidence import Evidence, Notebook
from rewind.core.ask.session import Session
from rewind.core.ask.watch import Watcher
from rewind.core.perception import PerceptionRunner
from rewind.core.ports import (
    BlobStore,
    CompletionRequest,
    LanguageModel,
    LanguageModelError,
    Segmenter,
    SystemMessage,
    Tool,
    UserMessage,
)

# A proof clip around one moment shows this long either side of it.
AROUND_S = 2.0
TILES_PER_SHEET = 9
# Most items one look covers; past this, the agent is told to ask about fewer.
MAX_LOOK_ITEMS = 45
# Items listed by name in a result; the count always covers all of them.
MAX_LISTED = 40
_LOOK_PROMPT_VERSION = "1"

LOOKING = (
    "You look closely at numbered tiles cut from video footage. Answer each tile from that "
    "tile only, in a few plain words. Say 'cannot tell' when the tile does not show it."
)
LOOK = "{boxed}Question for each tile: {question}"
LOOK_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tiles"],
    "properties": {
        "tiles": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["tile", "answer"],
                "properties": {"tile": {"type": "string"}, "answer": {"type": "string"}},
            },
        }
    },
}


class ToolError(Exception):
    """A problem the agent can fix by calling differently; the message goes back to it."""


@dataclass(frozen=True, slots=True)
class Item:
    # "the jar at 00:14": when it is seen largest, which is where its proof clip looks.
    label: str
    track: Track
    best: Observation


# What one question has found so far.
@dataclass(slots=True)
class Bench:
    session: Session
    model: LanguageModel
    segmenter: Segmenter
    notebook: Notebook
    found: dict[str, tuple[Item, ...]] = field(default_factory=dict[str, tuple[Item, ...]])
    # Every paid call of the question, recorded as it is made, so a failed step still counts.
    charges: list[Charge] = field(default_factory=list[Charge])

    @property
    def footage(self) -> Footage:
        return self.session.footage


@dataclass(frozen=True, slots=True)
class Outcome:
    evidence: tuple[Evidence, ...]
    # Shown as the finished step: "Found 26 jars".
    done: str

    # What the agent reads back: every finding with its id, nothing else.
    @property
    def text(self) -> str:
        return "\n".join(e.line for e in self.evidence) or "Nothing found."


@dataclass(frozen=True, slots=True)
class Started:
    # Shown while the tool runs: "Finding every jar".
    doing: str
    run: Callable[[Bench], Awaitable[Outcome]]


class AskTool(Protocol):
    @property
    def name(self) -> str: ...

    def as_tool(self) -> Tool: ...

    def start(self, raw_arguments: str) -> Started | str:
        """The run, or why the arguments are wrong."""
        ...


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True, slots=True)
class ToolSpec[A: Arguments]:
    name: str
    description: str
    arguments: type[A]
    doing: Callable[[A], str]
    run: Callable[[Bench, A], Awaitable[Outcome]]

    def as_tool(self) -> Tool:
        return Tool(self.name, self.description, self.arguments.model_json_schema())

    def start(self, raw_arguments: str) -> Started | str:
        try:
            arguments = self.arguments.model_validate_json(raw_arguments or "{}")
        except ValidationError as error:
            return f"invalid arguments for {self.name}: {error}"

        async def run(bench: Bench) -> Outcome:
            return await self.run(bench, arguments)

        return Started(self.doing(arguments), run)


class WatchArguments(Arguments):
    question: str = Field(min_length=1, description="What to look for, in plain words.")
    start_s: float | None = Field(None, description="Start of the stretch; omit for the start.")
    end_s: float | None = Field(None, description="End of the stretch; omit for the end.")


class FindArguments(Arguments):
    thing: str = Field(
        min_length=1, description="One kind of thing as a short noun phrase: 'jar', 'person'."
    )


class LookArguments(Arguments):
    question: str = Field(
        min_length=1, description="Asked of each item or frame: 'Does it have a label?'"
    )
    thing: str | None = Field(None, description="Look at items of this thing (runs find).")
    items: list[str] | None = Field(
        None, description="Only these items, named as found; omit for every item of the thing."
    )
    times: list[float] | None = Field(None, description="Or look at the whole frame at these.")


class MeasureArguments(Arguments):
    thing: str = Field(min_length=1, description="The thing to count and time (runs find).")


class Toolset:
    def __init__(
        self,
        blobs: BlobStore,
        watcher: Watcher,
        runner: PerceptionRunner,
        sheets: SheetMaker,
        track_fps: float,
    ) -> None:
        self._blobs = blobs
        self._watcher = watcher
        self._runner = runner
        self._sheets = sheets
        self._track_fps = track_fps

    def tools(self) -> tuple[AskTool, ...]:
        return (
            ToolSpec(
                "watch",
                "Muse watches the whole video or one stretch and answers in plain words, with "
                "the moments behind it. For what happens, when and why; not for exact counts.",
                WatchArguments,
                _watching,
                self.watch,
            ),
            ToolSpec(
                "find",
                "SAM finds and tracks every item of one kind of thing through the video and "
                "names each one by when it is seen best. Costs more than the others.",
                FindArguments,
                lambda a: f"Finding every {a.thing}",
                self.find,
            ),
            ToolSpec(
                "look",
                "Muse looks closely at each item of a found thing, boxed on its best frame, or "
                "at whole frames, and answers the same question for each one.",
                LookArguments,
                lambda a: f"Looking closer: {a.question}",
                self.look,
            ),
            ToolSpec(
                "measure",
                "Code counts the items of a thing and times them: the usual gap between one "
                "and the next, stops in the flow, and how long each one was in view.",
                MeasureArguments,
                lambda a: f"Measuring the {plural(a.thing, 2)}",
                self.measure,
            ),
        )

    async def watch(self, bench: Bench, arguments: WatchArguments) -> Outcome:
        duration = bench.footage.duration_s
        start = min(max(arguments.start_s or 0.0, 0.0), duration)
        end = min(max(arguments.end_s if arguments.end_s is not None else duration, 0), duration)
        if end - start < 1.0:
            raise ToolError("the stretch must be at least 1 s long and inside the video")
        span = TimeSpan(start=start, end=end)
        whole = start == 0 and end == duration
        watched, charges = await self._watcher.watch(
            bench.footage, span, arguments.question, bench.model
        )
        bench.charges += charges
        notebook = bench.notebook
        where = "In the whole video" if whole else f"From {_span_text(span)}"
        evidence = [
            notebook.add("seen", f"{where}: {watched.answer}", None if whole else span),
            *(notebook.add("seen", f"{_span_text(m.span)}: {m.what}", _around(m.span, duration))
              for m in watched.moments),
        ]  # fmt: skip
        return Outcome(tuple(evidence), f"Watched {_where(span, duration)}")

    async def find(self, bench: Bench, arguments: FindArguments) -> Outcome:
        thing = _thing(arguments.thing)
        items = await self.items(bench, thing)
        if not items:
            evidence = bench.notebook.add("measured", f"No {thing} was found.")
            return Outcome((evidence,), f"Found no {thing}")
        count = f"{len(items)} {plural(thing, len(items))}"
        evidence = bench.notebook.add("measured", f"{count} found: {_listed(items)}.")
        return Outcome((evidence,), f"Found {count}")

    async def look(self, bench: Bench, arguments: LookArguments) -> Outcome:
        footage = bench.footage
        subjects: list[tuple[str, Tile, Item | None]] = []
        if arguments.thing:
            thing = _thing(arguments.thing)
            items = await self.items(bench, thing)
            chosen = _chosen(items, arguments.items)
            subjects = [(i.label, Tile(i.best.t, i.best.box, ""), i) for i in chosen]
            what = f"{len(chosen)} {plural(thing, len(chosen))}"
            boxed = f"Each tile shows one {thing} boxed in red; answer about that {thing}.\n"
        elif arguments.times:
            times = sorted({min(max(t, 0.0), footage.duration_s) for t in arguments.times})
            subjects = [(f"the frame at {clock_time(t)}", Tile(t, None, ""), None) for t in times]
            what = f"{len(times)} {plural('frame', len(times))}"
            boxed = ""
        else:
            raise ToolError("give the thing whose items to look at, or the times of frames")
        if not subjects:
            raise ToolError("there is nothing to look at; find the thing first")
        if len(subjects) > MAX_LOOK_ITEMS:
            raise ToolError(f"that is {len(subjects)} items; ask about at most {MAX_LOOK_ITEMS}")
        evidence: list[Evidence] = []
        for first in range(0, len(subjects), TILES_PER_SHEET):
            batch = subjects[first : first + TILES_PER_SHEET]
            tiles = [Tile(tile.t, tile.box, str(n)) for n, (_, tile, _) in enumerate(batch, 1)]
            prompt = LOOK.format(boxed=boxed, question=arguments.question)
            answers = await self._ask_sheet(bench, tiles, prompt)
            for (label, tile, item), shown in zip(batch, tiles, strict=True):
                answer = answers.get(shown.label, "cannot tell")
                clip = _around(TimeSpan(start=tile.t, end=tile.t), footage.duration_s)
                marks = () if item is None else (_mark(item, clip, answer),)
                text = f"{label[0].upper()}{label[1:]}: {answer}"
                evidence.append(bench.notebook.add("seen", text, clip, marks))
        return Outcome(tuple(evidence), f"Looked closer at {what}")

    async def measure(self, bench: Bench, arguments: MeasureArguments) -> Outcome:
        thing = _thing(arguments.thing)
        items = await self.items(bench, thing)
        notebook = bench.notebook
        if not items:
            evidence = notebook.add("measured", f"No {thing} was found to measure.")
            return Outcome((evidence,), f"Found no {thing}")
        footage = bench.footage
        ends = [item.track.observations[-1].t for item in items]
        usual = usual_gap(ends)
        count = f"{len(items)} {plural(thing, len(items))}"
        pace = f"; usually {usual:.1f} s from one {thing} leaving to the next" if usual else ""
        evidence = [notebook.add("measured", f"{count} in the video{pace}.")]
        judged = [Judged(rule_id="measure", track_id=i.label, t=t)
                  for i, t in zip(items, ends, strict=True)]  # fmt: skip
        paths = [[(o.t, (o.box.x0 + o.box.x1) / 2, (o.box.y0 + o.box.y1) / 2)
                  for o in item.track.observations] for item in items]  # fmt: skip
        stops = find_stops(judged, bench.session.active_spans, footage.duration_s, paths)
        for stop in stops:
            still = (
                "nothing moved"
                if stop.seen == "no_motion"
                else f"the {plural(thing, 2)} stood still"
            )
            lost = f"about {stop.lost_items} {plural(thing, stop.lost_items)} fewer than usual"
            text = (
                f"Stop: no {thing} came through from {_span_text(stop.gap)}; {still} from "
                f"{_span_text(stop.stood)} ({stop.stood.end - stop.stood.start:.0f} s), {lost}."
            )
            evidence.append(notebook.add("measured", text, stop.stood))
        in_view = "; ".join(
            f"{item.label} {_span_text(_seen(item))} ({_length(_seen(item)):.0f} s)"
            for item in items[:MAX_LISTED]
        )
        evidence.append(notebook.add("measured", f"Time in view: {in_view}."))
        stops_text = f"{len(stops)} {plural('stop', len(stops))}"
        return Outcome(tuple(evidence), f"Measured {count}: {stops_text}")

    # SAM tracks the thing once over the stretches where something moves; the observation cache
    # makes asking again free.
    async def items(self, bench: Bench, thing: str) -> tuple[Item, ...]:
        if thing in bench.found:
            return bench.found[thing]
        footage = bench.footage
        spans = bench.session.active_spans or (TimeSpan(start=0, end=footage.duration_s),)
        plan = SamplingPlan(
            footage_id=footage.id,
            strategy="ask",
            windows=tuple(
                SamplingWindow(span=span, fps=self._track_fps, concepts=(thing,)) for span in spans
            ),
        )
        result = await self._runner.run(footage, plan, bench.segmenter)
        bench.charges += result.charges
        tracks = naive_tracks(footage.id, result.observations, max(1.0, 3 / self._track_fps))
        # One sighting is too little to tell an item from noise.
        items = _named(thing, [track for track in tracks if len(track.observations) >= 2])
        bench.found[thing] = items
        return items

    # One picture of up to nine tiles per call; kept by its inputs, so the same look is free.
    async def _ask_sheet(self, bench: Bench, tiles: Sequence[Tile], prompt: str) -> dict[str, str]:
        footage = bench.footage
        drawn = [(tile.t, tile.box.model_dump() if tile.box else None, tile.label)
                 for tile in tiles]  # fmt: skip
        digest = _digest([footage.content_hash, drawn])
        sheet_key = f"ask/sheets/{digest}.jpg"
        asked = _digest([digest, prompt, bench.model.model_version, _LOOK_PROMPT_VERSION])
        answer_key = f"ask/looked/{asked}.json"
        if await self._blobs.exists(answer_key):
            return _answers((await self._blobs.read(answer_key)).decode())
        if not await self._blobs.exists(sheet_key):
            source = await self._blobs.local_path(footage.original_key)
            with tempfile.TemporaryDirectory() as workdir:
                destination = Path(workdir) / "sheet.jpg"
                await self._sheets.make_sheet(source, destination, tiles)
                await self._blobs.write_file(sheet_key, destination)
        completion = await bench.model.complete(
            CompletionRequest(
                messages=(SystemMessage(LOOKING), UserMessage(prompt, image_keys=(sheet_key,))),
                max_output_tokens=2000,
                response_schema=LOOK_SCHEMA,
                effort="low",
            )
        )
        bench.charges += completion.charges
        answers = _answers(completion.text)
        await self._blobs.write(answer_key, completion.text.encode())
        return answers


class _TileAnswer(BaseModel):
    tile: str
    answer: str


class _SheetAnswer(BaseModel):
    tiles: list[_TileAnswer]


def _answers(text: str) -> dict[str, str]:
    try:
        parsed = _SheetAnswer.model_validate_json(text)
    except ValidationError as error:
        raise LanguageModelError(f"Muse's look did not fit its schema: {error}") from error
    return {t.tile.strip(): " ".join(t.answer.split()) for t in parsed.tiles if t.answer.strip()}


def _watching(arguments: WatchArguments) -> str:
    if arguments.start_s is None and arguments.end_s is None:
        return f"Watching the video: {arguments.question}"
    start = clock_time(arguments.start_s or 0.0)
    end = clock_time(arguments.end_s) if arguments.end_s is not None else "the end"
    return f"Watching {start} to {end}: {arguments.question}"


def _thing(text: str) -> str:
    return " ".join(text.lower().split())


# Each item is named by the moment it is seen largest. Two items at the same second are told
# apart by where they are, from the left.
def _named(thing: str, tracks: Sequence[Track]) -> tuple[Item, ...]:
    best = [max(t.observations, key=lambda o: (o.box.x1 - o.box.x0) * (o.box.y1 - o.box.y0))
            for t in tracks]  # fmt: skip
    groups: defaultdict[str, list[int]] = defaultdict(list)
    for index, observation in enumerate(best):
        groups[clock_time(observation.t)].append(index)
    labels: dict[int, str] = {}
    for clock, indices in groups.items():
        indices.sort(key=lambda i: best[i].box.x0 + best[i].box.x1)
        for place, index in enumerate(indices, 1):
            where = "" if len(indices) == 1 else f"{_ordinal(place)} from the left "
            labels[index] = f"the {thing} {where}at {clock}"
    order = sorted(range(len(tracks)), key=lambda i: (best[i].t, labels[i]))
    return tuple(Item(labels[i], tracks[i], best[i]) for i in order)


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _chosen(items: Sequence[Item], names: Sequence[str] | None) -> list[Item]:
    if not names:
        return list(items)
    by_label = {item.label: item for item in items}
    wanted = [" ".join(name.lower().split()) for name in names]
    unknown = [name for name in wanted if name not in by_label]
    if unknown:
        raise ToolError(f"no items named {join_words(unknown)}; use the names find gave")
    return [by_label[name] for name in dict.fromkeys(wanted)]


def _listed(items: Sequence[Item]) -> str:
    named = [item.label for item in items[:MAX_LISTED]]
    rest = len(items) - len(named)
    return join_words([*named, f"{rest} more"] if rest else named)


def _mark(item: Item, clip: TimeSpan, answer: str) -> Mark:
    path = tuple(
        Sighting(t=o.t, box=o.box)
        for o in item.track.observations
        if clip.start - 1 <= o.t <= clip.end + 1
    )
    tag = answer if len(answer) <= 24 else answer[:23] + "…"
    return Mark(kind="flagged", tag=tag, path=path or (Sighting(t=item.best.t, box=item.best.box),))


def _seen(item: Item) -> TimeSpan:
    return TimeSpan(start=item.track.observations[0].t, end=item.track.observations[-1].t)


def _length(span: TimeSpan) -> float:
    return span.end - span.start


def _around(span: TimeSpan, duration_s: float) -> TimeSpan:
    return TimeSpan(start=max(0.0, span.start - AROUND_S), end=min(duration_s, span.end + AROUND_S))


def _span_text(span: TimeSpan) -> str:
    return f"{clock_time(span.start)} to {clock_time(span.end)}"


def _where(span: TimeSpan, duration_s: float) -> str:
    if span.start == 0 and span.end == duration_s:
        return "the whole video"
    return _span_text(span)


def _digest(parts: Sequence[object]) -> str:
    return hashlib.sha256(json.dumps(parts, default=str).encode()).hexdigest()[:24]
