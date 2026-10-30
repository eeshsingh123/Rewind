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
    Box,
    Charge,
    Footage,
    Observation,
    SamplingPlan,
    SamplingWindow,
    TimeSpan,
    Track,
)
from rewind.core.analysis import (
    PartLooker,
    SheetMaker,
    Sighting,
    Tile,
    clock_time,
    join_words,
    naive_tracks,
)
from rewind.core.analysis.proof import Mark
from rewind.core.analysis.replies import plural
from rewind.core.analysis.timeline import Judged, find_stops, usual_gap
from rewind.core.ask.areas import Area, AreaMarker, short_tag
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
_LOOK_PROMPT_VERSION = "2"

LOOKING = (
    "You look closely at numbered tiles cut from a photo or a video. Answer each tile from that "
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
    # When it is seen largest, which is where its proof looks; on a photo, where it is.
    label: str
    track: Track
    best: Observation


# Video tracks items over time; image looks at one frame, a photo or a frame to mark an area on.
@dataclass(frozen=True, slots=True)
class Segmenters:
    video: Segmenter
    image: Segmenter


# What one question has found so far.
@dataclass(slots=True)
class Bench:
    session: Session
    model: LanguageModel
    segmenters: Segmenters
    notebook: Notebook
    found: dict[str, tuple[Item, ...]] = field(default_factory=dict[str, tuple[Item, ...]])
    areas: dict[str, Area] = field(default_factory=dict[str, Area])
    # Every paid call of the question, recorded as it is made, so a failed step still counts.
    charges: list[Charge] = field(default_factory=list[Charge])

    @property
    def footage(self) -> Footage:
        return self.session.footage

    @property
    def media(self) -> str:
        return "photo" if self.session.photo else "video"


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

    # Not offered on a photo.
    @property
    def video_only(self) -> bool: ...

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
    video_only: bool = False

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
        min_length=1, description="One kind of thing as a short noun phrase: 'chair', 'person'."
    )
    area: str | None = Field(None, description="Only in this area, named as area named it.")


class LookArguments(Arguments):
    question: str = Field(
        min_length=1, description="Asked of each item or frame: 'Does it have a label?'"
    )
    thing: str | None = Field(None, description="Look at items of this thing (runs find).")
    items: list[str] | None = Field(
        None, description="Only these items, named as found; omit for every item of the thing."
    )
    times: list[float] | None = Field(
        None, description="Or look at the whole frame at these times; a photo needs none."
    )
    area: str | None = Field(
        None,
        description="Only in this area, named as area named it. Without a thing, look at the "
        "area itself.",
    )


class MeasureArguments(Arguments):
    thing: str = Field(min_length=1, description="The thing to count and time (runs find).")
    area: str | None = Field(None, description="Only in this area, named as area named it.")


class AreaArguments(Arguments):
    name: str = Field(
        min_length=1, description="The part of the frame, in plain words: 'the left half'."
    )
    at_s: float | None = Field(None, description="The time of the frame; a photo needs none.")


class Toolset:
    def __init__(
        self,
        blobs: BlobStore,
        watcher: Watcher,
        runner: PerceptionRunner,
        stills: PartLooker,
        sheets: SheetMaker,
        track_fps: float,
    ) -> None:
        self._blobs = blobs
        self._watcher = watcher
        self._runner = runner
        self._stills = stills
        self._marker = AreaMarker(blobs, stills)
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
                video_only=True,
            ),
            ToolSpec(
                "find",
                "SAM finds every item of one kind of thing, in a photo or through a video, and "
                "names each one: by where it is on a photo, by when it is seen best in a video. "
                "Costs more than the others.",
                FindArguments,
                lambda a: f"Finding every {a.thing}",
                self.find,
            ),
            ToolSpec(
                "look",
                "Muse looks closely at each item of a found thing, boxed on its best frame, or "
                "at a whole frame or area, and answers the same question for each one.",
                LookArguments,
                lambda a: f"Looking closer: {a.question}",
                self.look,
            ),
            ToolSpec(
                "measure",
                "Code counts the items of a thing. In a video it also times them: the usual gap "
                "between one and the next, stops in the flow, and how long each was in view.",
                MeasureArguments,
                lambda a: f"Counting the {plural(a.thing, 2)}",
                self.measure,
            ),
            ToolSpec(
                "area",
                "Marks one named part of the frame, so find, look and measure can work on that "
                "part alone. Muse marks it and SAM confirms it; if they cannot, it says so.",
                AreaArguments,
                lambda a: f"Finding {_thing(a.name)}",
                self.area,
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
        area = _area(bench, arguments.area)
        items = _within(await self.items(bench, thing), area)
        place = _in(area)
        clip, marks = _count_proof(bench, items, area)
        if not items:
            evidence = bench.notebook.add("measured", f"No {thing} was found{place}.", clip, marks)
            return Outcome((evidence,), f"Found no {thing}{place}")
        count = f"{len(items)} {plural(thing, len(items))}"
        text = f"{count} found{place}: {_listed(items)}."
        evidence = bench.notebook.add("measured", text, clip, marks)
        return Outcome((evidence,), f"Found {count}{place}")

    async def look(self, bench: Bench, arguments: LookArguments) -> Outcome:
        footage = bench.footage
        area = _area(bench, arguments.area)
        subjects: list[tuple[str, Tile, Item | None]] = []
        if arguments.thing:
            thing = _thing(arguments.thing)
            items = _within(await self.items(bench, thing), area)
            chosen = _chosen(items, arguments.items)
            subjects = [(i.label, Tile(i.best.t, i.best.box, ""), i) for i in chosen]
            what = f"{len(chosen)} {plural(thing, len(chosen))}"
            boxed = f"Each tile shows one {thing} boxed in red; answer about that {thing}.\n"
        else:
            times = sorted(_look_times(bench, arguments.times, area))
            box = area.box if area else None
            subjects = [(_frame_name(bench, t, area), Tile(t, box, ""), None) for t in times]
            whole = "the photo" if bench.session.photo else f"{len(times)} frames"
            what = area.name if area else whole
            boxed = ""
            if area:
                boxed = f"Each tile shows {area.name} boxed in red; answer about inside the box.\n"
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
                marks = (
                    *_area_marks(area),
                    *(() if item is None else (_mark(item, clip, answer),)),
                )
                text = f"{label[0].upper()}{label[1:]}: {answer}"
                evidence.append(bench.notebook.add("seen", text, clip, marks))
        return Outcome(tuple(evidence), f"Looked closer at {what}")

    async def measure(self, bench: Bench, arguments: MeasureArguments) -> Outcome:
        thing = _thing(arguments.thing)
        area = _area(bench, arguments.area)
        items = _within(await self.items(bench, thing), area)
        notebook = bench.notebook
        place = _in(area)
        clip, marks = _count_proof(bench, items, area)
        if not items:
            text = f"No {thing} was found to measure{place}."
            return Outcome((notebook.add("measured", text, clip, marks),), f"Found no {thing}")
        count = f"{len(items)} {plural(thing, len(items))}"
        place = place or f" in the {bench.media}"
        if bench.session.photo:
            evidence = notebook.add("measured", f"{count}{place}.", clip, marks)
            return Outcome((evidence,), f"Counted {count}{place}")
        footage = bench.footage
        ends = [item.track.observations[-1].t for item in items]
        usual = usual_gap(ends)
        pace = f"; usually {usual:.1f} s from one {thing} leaving to the next" if usual else ""
        evidence = [notebook.add("measured", f"{count}{place}{pace}.", clip, marks)]
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

    async def area(self, bench: Bench, arguments: AreaArguments) -> Outcome:
        name = _thing(arguments.name)
        footage = bench.footage
        if bench.session.photo:
            t, frame = 0.0, "the photo"
        elif arguments.at_s is None:
            raise ToolError("give the time of the frame to mark the area on")
        else:
            t = min(max(arguments.at_s, 0.0), footage.duration_s)
            frame = f"the frame at {clock_time(t)}"
        area = await self._marker.mark(
            footage, name, t, bench.model, bench.segmenters.image, bench.charges
        )
        if area is None:
            evidence = bench.notebook.add("seen", f"Could not make out {name} on {frame}.")
            return Outcome((evidence,), f"Could not make out {name}")
        bench.areas[name] = area
        clip = _around(TimeSpan(start=t, end=t), footage.duration_s)
        text = f"{name[0].upper()}{name[1:]} is marked on {frame}."
        evidence = bench.notebook.add("seen", text, clip, (area.mark,))
        return Outcome((evidence,), f"Marked {name}")

    async def items(self, bench: Bench, thing: str) -> tuple[Item, ...]:
        if thing in bench.found:
            return bench.found[thing]
        look = self._on_photo if bench.session.photo else self._tracked
        items = await look(bench, thing)
        bench.found[thing] = items
        return items

    # One image call finds them all, however many; the still cache makes asking again free.
    async def _on_photo(self, bench: Bench, thing: str) -> tuple[Item, ...]:
        footage = bench.footage
        looks = await self._stills.look(footage, {thing: {0.0}}, bench.segmenters.image)
        bench.charges += looks.charges
        tracks = [
            Track(footage_id=footage.id, track_id=f"{thing}-{n}", concept=thing, observations=(o,))
            for n, o in enumerate(looks.observations, 1)
        ]
        return _placed(thing, tracks)

    # SAM tracks the thing once over the stretches where something moves; the observation cache
    # makes asking again free.
    async def _tracked(self, bench: Bench, thing: str) -> tuple[Item, ...]:
        footage = bench.footage
        spans = bench.session.active_spans or (TimeSpan(start=0, end=footage.duration_s),)
        plan = SamplingPlan(
            footage_id=footage.id,
            strategy="ask",
            windows=tuple(
                SamplingWindow(span=span, fps=self._track_fps, concepts=(thing,)) for span in spans
            ),
        )
        result = await self._runner.run(footage, plan, bench.segmenters.video)
        bench.charges += result.charges
        tracks = naive_tracks(footage.id, result.observations, max(1.0, 3 / self._track_fps))
        # One sighting is too little to tell an item from noise.
        return _named(thing, [track for track in tracks if len(track.observations) >= 2])

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
    best = [_largest(t.observations) for t in tracks]
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


# On a photo an item has only its place: its order from the left and, when items stand in rows,
# its row. An item joins a row when its middle is above the bottom of the row's first item.
def _placed(thing: str, tracks: Sequence[Track]) -> tuple[Item, ...]:
    boxes = [track.observations[0].box for track in tracks]
    rows: list[list[int]] = []
    for index in sorted(range(len(tracks)), key=lambda i: _middle_y(boxes[i])):
        if rows and _middle_y(boxes[index]) < boxes[rows[-1][0]].y1:
            rows[-1].append(index)
        else:
            rows.append([index])
    items: list[Item] = []
    for number, row in enumerate(rows):
        row.sort(key=lambda i: (boxes[i].x0 + boxes[i].x1, boxes[i].y0))
        in_row = f" in {_row_name(number, len(rows))}" if len(rows) > 1 else ""
        for place, index in enumerate(row, 1):
            name = (
                f"the {_ordinal(place)} {thing} from the left" if len(row) > 1 else f"the {thing}"
            )
            track = tracks[index]
            items.append(Item(f"{name}{in_row}", track, track.observations[0]))
    return tuple(items)


def _row_name(number: int, rows: int) -> str:
    if number == 0:
        return "the top row"
    if number == rows - 1:
        return "the bottom row"
    return f"the {_ordinal(number + 1)} row from the top"


def _middle_y(box: Box) -> float:
    return (box.y0 + box.y1) / 2


def _largest(observations: Sequence[Observation]) -> Observation:
    return max(observations, key=lambda o: (o.box.x1 - o.box.x0) * (o.box.y1 - o.box.y0))


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


def _area(bench: Bench, name: str | None) -> Area | None:
    if name is None:
        return None
    area = bench.areas.get(_thing(name))
    if area is None:
        raise ToolError(f"no area named {name}; mark it first with the area tool")
    return area


# Only the sightings inside the area count, so an item's times there are its own.
def _within(items: Sequence[Item], area: Area | None) -> tuple[Item, ...]:
    if area is None:
        return tuple(items)
    kept: list[Item] = []
    for item in items:
        inside = tuple(o for o in item.track.observations if area.holds(o.box))
        if inside:
            track = item.track.model_copy(update={"observations": inside})
            kept.append(Item(item.label, track, _largest(inside)))
    return tuple(kept)


def _in(area: Area | None) -> str:
    return f" in {area.name}" if area else ""


def _area_marks(area: Area | None) -> tuple[Mark, ...]:
    return (area.mark,) if area else ()


def _look_times(bench: Bench, times: Sequence[float] | None, area: Area | None) -> set[float]:
    if bench.session.photo:
        return {0.0}
    if times:
        return {min(max(t, 0.0), bench.footage.duration_s) for t in times}
    if area:
        return {area.t}
    raise ToolError("give the thing whose items to look at, the times of frames, or an area")


def _frame_name(bench: Bench, t: float, area: Area | None) -> str:
    if bench.session.photo:
        return area.name if area else "the photo"
    return f"{area.name if area else 'the frame'} at {clock_time(t)}"


# A count has something to show on a photo or in an area: every counted item, boxed. A count over
# a whole video has no one moment.
def _count_proof(
    bench: Bench, items: Sequence[Item], area: Area | None
) -> tuple[TimeSpan | None, tuple[Mark, ...]]:
    if area is None and not bench.session.photo:
        return None, ()
    t = area.t if area else 0.0
    clip = _around(TimeSpan(start=t, end=t), bench.footage.duration_s)
    paths = [_path(item, clip) for item in items]
    counted = [Mark(kind="passed", tag="", path=path) for path in paths if path]
    return clip, (*_area_marks(area), *counted)


def _path(item: Item, clip: TimeSpan) -> tuple[Sighting, ...]:
    return tuple(
        Sighting(t=o.t, box=o.box)
        for o in item.track.observations
        if clip.start - 1 <= o.t <= clip.end + 1
    )


def _mark(item: Item, clip: TimeSpan, answer: str) -> Mark:
    path = _path(item, clip) or (Sighting(t=item.best.t, box=item.best.box),)
    return Mark(kind="flagged", tag=short_tag(answer), path=path)


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
