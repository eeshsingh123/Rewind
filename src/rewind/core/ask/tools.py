import asyncio
import hashlib
import json
import logging
import tempfile
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

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
from rewind.core.ask.areas import Area, AreaMarker, short_tag
from rewind.core.ask.evidence import Evidence, Notebook
from rewind.core.ask.marks import Mark, Sighting
from rewind.core.ask.names import MAX_NAMES, NAME, NAMES_SCHEMA, NAMING, chosen_names, names_from
from rewind.core.ask.session import Marked, OnStep, Session, Step
from rewind.core.ask.sheets import SheetMaker, Tile
from rewind.core.ask.stops import find_stops, usual_gap
from rewind.core.ask.watch import Moment, Watcher
from rewind.core.ask.words import clock_time, join_words, plural
from rewind.core.perception import (
    WINDOW_S,
    PartLooker,
    PerceptionRunner,
    distinct_tracks,
    naive_tracks,
)
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

logger = logging.getLogger(__name__)

# A proof clip around one moment shows this long either side of it.
AROUND_S = 2.0
BROAD_SHARE = 0.8
TILES_PER_SHEET = 9
# Close looks sent to Muse at the same time.
LOOKS_AT_ONCE = 4
# Most items drawn on one combined proof.
MAX_PROOF_TILES = 12
# A second view of an item must show it at least this large, compared with its best view.
SECOND_VIEW_SIZE = 0.5
# Timed live on a 20 s clip at 2 fps: about 2 min, for one name or two tracked side by side.
SAM_S_PER_FRAME = 2.5
# Most items one look covers; past this, the agent is told to ask about fewer.
MAX_LOOK_ITEMS = 45
# What follow names the thing a person dragged a box around; look then takes it as a thing.
MARKED_THING = "marked thing"
_FOLLOW_SHOWN_S = 10.0
# Items listed by name in a result; the count always covers all of them.
MAX_LISTED = 40
_LOOK_PROMPT_VERSION = "3"
_NAME_PROMPT_VERSION = "1"
# Stills each name is tried on, and that Muse sees to name the thing.
NAME_MOMENTS = 3
CONFIRM = "Each tile shows one thing boxed in red, found by a search.\n"
CONFIRM_QUESTION = (
    "Could a person fairly call what is boxed one {}? Say no only when it is clearly something "
    "else, such as part of a wall or the background, or only a piece of a bigger object."
)
PLACE = "Each tile shows one item boxed in red, with what is around it.\n"
PLACE_QUESTION = (
    "Is the boxed item in {0}? Say no only when it clearly is not, such as when it stands behind "
    "or in front of {0} without touching it. When the tile cannot show that, say cannot tell."
)

type Verdict = Literal["yes", "no", "cannot tell", "other"]

LOOKING = (
    "You look closely at numbered tiles cut from a photo or a video. Answer each tile from that "
    "tile only. verdict: yes or no to the question; 'cannot tell' when the tile does not show "
    "it; 'other' when it is not a yes or no question. answer: a few plain words that make sense "
    "alone, such as 'no label' or 'flap open to the right', never just yes or no."
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
                "required": ["tile", "verdict", "answer"],
                "properties": {
                    "tile": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["yes", "no", "cannot tell", "other"]},
                    "answer": {"type": "string"},
                },
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


# One item's answer to one close-look question, and the view it was judged on.
@dataclass(frozen=True, slots=True)
class Judged:
    item: Item
    verdict: Verdict
    tile: Tile


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
    marked: Marked | None = None
    # Each close-look question's answers, by item, for combine.
    judged: dict[str, dict[str, Judged]] = field(default_factory=dict[str, dict[str, Judged]])
    on_step: OnStep | None = None

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

    # Offered only when the question came with a dragged box.
    @property
    def needs_mark(self) -> bool: ...

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
    needs_mark: bool = False

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
        min_length=1,
        description="One kind of thing as a plain noun, with no place or condition: 'chair'. "
        "Use area for a place and look for a condition.",
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


class FollowArguments(Arguments):
    pass


class Condition(Arguments):
    question: str = Field(min_length=1, description="A look question, word for word.")
    want: Literal["yes", "no"] = Field(description="The answer an item must have to match.")


class CombineArguments(Arguments):
    conditions: list[Condition] = Field(min_length=1)
    how: Literal["all", "any"] = Field(
        "all", description="Must an item meet all the conditions, or any one of them?"
    )
    describe: str = Field(min_length=1, description="What a match is: 'boxes with no label'.")


class _TileAnswer(BaseModel):
    tile: str
    verdict: Verdict = "cannot tell"
    answer: str


type _Subject = tuple[str, Tile, Item | None]


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
                "combine",
                "Code combines the answers look gave, item by item, into the exact list and "
                "count of the items that match, and those it could not tell. Call it once for "
                "each list the person wants.",
                CombineArguments,
                lambda a: f"Listing {a.describe}",
                self.combine,
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
            ToolSpec(
                "follow",
                "SAM follows what the person marked through the whole video. Says when it is "
                "in view and where it goes, with a clip.",
                FollowArguments,
                lambda _: "Following what you marked",
                self.follow,
                video_only=True,
                needs_mark=True,
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
            *(note_moment(notebook, m, duration, _around(m.span, duration))
              for m in watched.moments),
        ]  # fmt: skip
        return Outcome(tuple(evidence), f"Watched {_where(span, duration)}")

    async def find(self, bench: Bench, arguments: FindArguments) -> Outcome:
        thing = _known_thing(bench, arguments.thing)
        area = _area(bench, arguments.area)
        items = await self._inside(bench, thing, area)
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
            thing = _known_thing(bench, arguments.thing)
            items = await self._inside(bench, thing, area)
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
        prompt = LOOK.format(boxed=boxed, question=arguments.question)
        answers = await self._ask_tiles(bench, [tile for _, tile, _ in subjects], prompt)
        subjects, answers = await self._second_views(bench, subjects, answers, prompt)
        asked = bench.judged.setdefault(_thing(arguments.question), {})
        evidence: list[Evidence] = []
        for (label, tile, item), answer in zip(subjects, answers, strict=True):
            clip = _around(TimeSpan(start=tile.t, end=tile.t), footage.duration_s)
            tag = _tag(answer, arguments.question)
            marks = (*_area_marks(area), *(() if item is None else (_mark(item, clip, tag),)))
            text = f"{label[0].upper()}{label[1:]}: {answer.answer} ({answer.verdict})"
            evidence.append(bench.notebook.add("seen", text, clip, marks))
            if item is not None:
                asked[item.label] = Judged(item, answer.verdict, tile)
        return Outcome(tuple(evidence), f"Looked closer at {what}")

    # An item that could not be judged on its best view gets one more, from another moment.
    async def _second_views(
        self, bench: Bench, subjects: list[_Subject], answers: list[_TileAnswer], prompt: str
    ) -> tuple[list[_Subject], list[_TileAnswer]]:
        again: list[tuple[int, _Subject]] = []
        for index, (label, _, item) in enumerate(subjects):
            view = None if item is None else _second_view(item)
            if view is not None and answers[index].verdict == "cannot tell":
                again.append((index, (label, Tile(view.t, view.box, ""), item)))
        if not again:
            return subjects, answers
        seen = await self._ask_tiles(bench, [subject[1] for _, subject in again], prompt)
        subjects, answers = list(subjects), list(answers)
        for (index, subject), answer in zip(again, seen, strict=True):
            if answer.verdict != "cannot tell":
                subjects[index], answers[index] = subject, answer
        return subjects, answers

    # Sheets go to Muse side by side: one after another, a clip's look took minutes.
    async def _ask_tiles(
        self, bench: Bench, tiles: Sequence[Tile], prompt: str
    ) -> list[_TileAnswer]:
        batches = [tiles[i : i + TILES_PER_SHEET] for i in range(0, len(tiles), TILES_PER_SHEET)]
        sheets = [[Tile(t.t, t.box, str(n)) for n, t in enumerate(b, 1)] for b in batches]
        gate = asyncio.Semaphore(LOOKS_AT_ONCE)

        async def ask(sheet: list[Tile]) -> dict[str, _TileAnswer]:
            async with gate:
                return await self._ask_sheet(bench, sheet, prompt)

        answered = await asyncio.gather(*(ask(sheet) for sheet in sheets))
        return [
            found.get(tile.label, _TileAnswer(tile=tile.label, answer="cannot tell"))
            for sheet, found in zip(sheets, answered, strict=True)
            for tile in sheet
        ]

    async def combine(self, bench: Bench, arguments: CombineArguments) -> Outcome:
        answers = [_answers_to(bench, c.question) for c in arguments.conditions]
        wanted = [c.want for c in arguments.conditions]
        labels = list(dict.fromkeys(label for asked in answers for label in asked))
        results: list[tuple[Judged, bool | None]] = []
        for label in labels:
            got = [asked.get(label) for asked in answers]
            states = [_meets(g, want) for g, want in zip(got, wanted, strict=True)]
            judged = next(g for g in got if g is not None)
            results.append((judged, _together(states, arguments.how)))
        matched = [judged for judged, state in results if state]
        unsure = [judged.item.label for judged, state in results if state is None]
        count = f"{arguments.describe[0].upper()}{arguments.describe[1:]}: "
        count += f"{len(matched)} of {len(results)}"
        listed = join_words([j.item.label for j in matched[:MAX_LISTED]])
        text = f"{count}: {listed}." if matched else f"{count}."
        if unsure:
            text += f" {len(unsure)} could not be told: {join_words(unsure[:MAX_LISTED])}."
        # With no match the proof shows what was checked.
        shown = (matched or [judged for judged, _ in results])[:MAX_PROOF_TILES]
        sheet = await self._proof_sheet(bench, shown)
        evidence = bench.notebook.add("seen", text, proof_key=sheet)
        return Outcome((evidence,), f"{len(matched)} of {len(results)}: {arguments.describe}")

    async def _proof_sheet(self, bench: Bench, shown: Sequence[Judged]) -> str:
        photo = bench.session.photo
        tiles = [
            Tile(j.tile.t, j.tile.box, str(n) if photo else clock_time(j.tile.t))
            for n, j in enumerate(shown, 1)
        ]
        return await self._sheet(bench, tiles)

    async def measure(self, bench: Bench, arguments: MeasureArguments) -> Outcome:
        thing = _known_thing(bench, arguments.thing)
        area = _area(bench, arguments.area)
        items = await self._inside(bench, thing, area)
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
        paths = [[(o.t, (o.box.x0 + o.box.x1) / 2, (o.box.y0 + o.box.y1) / 2)
                  for o in item.track.observations] for item in items]  # fmt: skip
        stops = find_stops(ends, bench.session.active_spans, footage.duration_s, paths)
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
        clip = _around(TimeSpan(start=t, end=t), footage.duration_s)
        area = await self._marker.mark(
            footage, name, t, bench.model, bench.segmenters.image, bench.charges
        )
        if area is None:
            evidence = bench.notebook.add("seen", f"Could not make out {name} on {frame}.", clip)
            return Outcome((evidence,), f"Could not make out {name}")
        bench.areas[name] = area
        text = f"{name[0].upper()}{name[1:]} is marked on {frame}."
        evidence = bench.notebook.add("seen", text, clip, (area.mark,))
        return Outcome((evidence,), f"Marked {name}")

    async def follow(self, bench: Bench, arguments: FollowArguments) -> Outcome:
        marked = bench.marked
        if marked is None:
            raise ToolError("nothing was marked")
        footage = bench.footage
        result = await self._runner.follow(
            footage, marked.t, marked.box, self._track_fps, bench.segmenters.video, MARKED_THING
        )
        bench.charges += result.charges
        seen = sorted(result.observations, key=lambda o: o.t)
        if not seen:
            clip = _around(TimeSpan(start=marked.t, end=marked.t), footage.duration_s)
            evidence = bench.notebook.add("seen", "What was marked could not be followed.", clip)
            return Outcome((evidence,), "Could not follow what you marked")
        track = Track(
            footage_id=footage.id,
            track_id=MARKED_THING,
            concept=MARKED_THING,
            observations=tuple(seen),
        )
        item = Item(f"the {MARKED_THING}", track, _largest(seen))
        bench.found[MARKED_THING] = (item,)
        span = _seen(item)
        path = tuple(Sighting(t=o.t, box=o.box) for o in seen)
        text = (
            f"The {MARKED_THING} is in view from {_span_text(span)} ({_length(span):.0f} s); "
            f"{_route(seen[0].box, seen[-1].box)}."
        )
        mark = Mark(kind="flagged", tag="followed", path=path)
        # A long track shows the stretch around where it was marked, not its start.
        shown = TimeSpan(
            start=max(span.start, marked.t - _FOLLOW_SHOWN_S),
            end=min(span.end, marked.t + _FOLLOW_SHOWN_S),
        )
        evidence = bench.notebook.add("measured", text, shown, (mark,))
        return Outcome((evidence,), f"Followed it from {_span_text(span)}")

    async def items(self, bench: Bench, thing: str) -> tuple[Item, ...]:
        if thing in bench.found:
            return bench.found[thing]
        photo = bench.session.photo
        moments = (0.0,) if photo else _moments(_spans(bench))
        names = await self._names(bench, thing, moments)
        look = self._on_photo if photo else self._tracked
        tracks = await self._real(bench, names, await look(bench, thing, names, moments))
        items = _placed(thing, tracks) if photo else _named(thing, tracks)
        bench.found[thing] = items
        return items

    # An area is a rectangle, so what stands behind or in front of a belt or a shelf falls inside
    # it. Muse sees each item where it is; it stays unless Muse is sure it is not in that place.
    async def _inside(self, bench: Bench, thing: str, area: Area | None) -> tuple[Item, ...]:
        items = _within(await self.items(bench, thing), area)
        if area is None or not items:
            return items
        prompt = LOOK.format(boxed=PLACE, question=PLACE_QUESTION.format(area.name))
        tiles = [Tile(item.best.t, item.best.box, "") for item in items]
        answers = await self._ask_tiles(bench, tiles, prompt)
        kept = zip(items, answers, strict=True)
        return tuple(item for item, answer in kept if answer.verdict != "no")

    # What to call the thing is Muse's to say, from the footage itself: the words a person uses
    # may be ones the search does not know, or fit only some of the things they mean.
    async def _names(self, bench: Bench, thing: str, moments: Sequence[float]) -> tuple[str, ...]:
        footage, model = bench.footage, bench.model
        asked = _digest([footage.content_hash, thing, model.model_version, _NAME_PROMPT_VERSION])
        answer_key = f"ask/names/{asked}.json"
        if await self._blobs.exists(answer_key):
            return names_from((await self._blobs.read(answer_key)).decode(), thing)
        tiles = [Tile(t, None, str(n)) for n, t in enumerate(moments, 1)]
        prompt = NAME.format(thing=thing, most=MAX_NAMES)
        request = CompletionRequest(
            messages=(
                SystemMessage(NAMING),
                UserMessage(prompt, image_keys=(await self._sheet(bench, tiles),)),
            ),
            max_output_tokens=2000,
            response_schema=NAMES_SCHEMA,
            effort="low",
        )
        try:
            completion = await model.complete(request)
        except LanguageModelError:
            logger.warning("%s: no other names for %r", footage.id, thing)
            return (thing,)
        bench.charges += completion.charges
        await self._blobs.write(answer_key, completion.text.encode())
        return names_from(completion.text, thing)

    # Muse looks once at each item, as the search also finds things that only look the part: a
    # sign on a wall, a corner of something bigger. It stays unless Muse is sure it is not one.
    async def _real(
        self, bench: Bench, names: Sequence[str], tracks: Sequence[Track]
    ) -> list[Track]:
        if not tracks:
            return []
        if bench.on_step is not None:
            await bench.on_step(Step(text=f"Checking each {names[0]}", done=False))
        best = [_largest(track.observations) for track in tracks]
        prompt = LOOK.format(boxed=CONFIRM, question=CONFIRM_QUESTION.format(" or ".join(names)))
        answers = await self._ask_tiles(bench, [Tile(o.t, o.box, "") for o in best], prompt)
        kept = zip(tracks, answers, strict=True)
        return [track for track, answer in kept if answer.verdict != "no"]

    # One image call per name finds them all, however many; the still cache makes it free again.
    async def _on_photo(
        self, bench: Bench, thing: str, names: Sequence[str], moments: Sequence[float]
    ) -> list[Track]:
        footage = bench.footage
        wanted = {name: set(moments) for name in names}
        looks = await self._stills.look(footage, wanted, bench.segmenters.image)
        bench.charges += looks.charges
        tracks = [
            Track(footage_id=footage.id, track_id=f"{thing}-{n}", concept=thing, observations=(o,))
            for n, o in enumerate(looks.observations, 1)
        ]
        return list(distinct_tracks(tracks, 0.0))

    # Each name is tried on a few stills, which costs little against a pass over the video, and
    # SAM tracks the ones worth it over the stretches where something moves. The observation
    # cache makes asking again free.
    async def _tracked(
        self, bench: Bench, thing: str, names: Sequence[str], moments: Sequence[float]
    ) -> list[Track]:
        footage = bench.footage
        spans = _spans(bench)
        video = bench.segmenters.video
        if bench.on_step is not None and thing not in bench.session.things:
            frames = sum(_length(span) for span in spans) * self._track_fps
            wait = _wait_words(frames * SAM_S_PER_FRAME)
            await bench.on_step(Step(text=f"Tracking every {thing}, {wait}", done=False))
        if len(names) > 1:
            wanted = {name: set(moments) for name in names}
            looks = await self._stills.look(footage, wanted, bench.segmenters.image)
            bench.charges += looks.charges
            seen: defaultdict[str, list[Observation]] = defaultdict(list)
            for observation in looks.observations:
                seen[observation.concept].append(observation)
            names = chosen_names(names, seen, video.max_tracked_objects)
        plan = SamplingPlan(
            footage_id=footage.id,
            windows=tuple(
                SamplingWindow(span=span, fps=self._track_fps, concepts=tuple(names))
                for span in spans
            ),
        )
        result = await self._runner.run(footage, plan, video)
        bench.charges += result.charges
        tracks = naive_tracks(footage.id, result.observations, max(1.0, 3 / self._track_fps))
        # A request can miss something for a whole window and notice it again in the next.
        distinct = distinct_tracks(tracks, 2 * WINDOW_S)
        # One sighting is too little to tell an item from noise.
        return [track for track in distinct if len(track.observations) >= 2]

    # One picture of up to nine tiles per call; kept by its inputs, so the same look is free.
    async def _ask_sheet(
        self, bench: Bench, tiles: Sequence[Tile], prompt: str
    ) -> dict[str, _TileAnswer]:
        digest = _sheet_digest(bench, tiles)
        asked = _digest([digest, prompt, bench.model.model_version, _LOOK_PROMPT_VERSION])
        answer_key = f"ask/looked/{asked}.json"
        if await self._blobs.exists(answer_key):
            return _answers((await self._blobs.read(answer_key)).decode())
        sheet_key = await self._sheet(bench, tiles)
        completion = await bench.model.complete(
            CompletionRequest(
                messages=(SystemMessage(LOOKING), UserMessage(prompt, image_keys=(sheet_key,))),
                max_output_tokens=4000,
                response_schema=LOOK_SCHEMA,
                effort="low",
            )
        )
        bench.charges += completion.charges
        answers = _answers(completion.text)
        await self._blobs.write(answer_key, completion.text.encode())
        return answers

    async def _sheet(self, bench: Bench, tiles: Sequence[Tile]) -> str:
        sheet_key = f"ask/sheets/{_sheet_digest(bench, tiles)}.jpg"
        if not await self._blobs.exists(sheet_key):
            source = await self._blobs.local_path(bench.footage.original_key)
            with tempfile.TemporaryDirectory() as workdir:
                destination = Path(workdir) / "sheet.jpg"
                await self._sheets.make_sheet(source, destination, tiles)
                await self._blobs.write_file(sheet_key, destination)
        return sheet_key


class _SheetAnswer(BaseModel):
    tiles: list[_TileAnswer]


def _sheet_digest(bench: Bench, tiles: Sequence[Tile]) -> str:
    drawn = [(tile.t, tile.box.model_dump() if tile.box else None, tile.label) for tile in tiles]
    return _digest([bench.footage.content_hash, drawn])


def _answers(text: str) -> dict[str, _TileAnswer]:
    try:
        parsed = _SheetAnswer.model_validate_json(text)
    except ValidationError as error:
        raise LanguageModelError(f"Muse's look did not fit its schema: {error}") from error
    return {
        t.tile.strip(): t.model_copy(update={"answer": " ".join(t.answer.split())})
        for t in parsed.tiles
        if t.answer.strip()
    }


# The answers look gave to a question; the agent may vary its case and spacing.
def _answers_to(bench: Bench, question: str) -> dict[str, Judged]:
    asked = bench.judged.get(_thing(question))
    if asked is None:
        known = join_words([f'"{q}"' for q in bench.judged]) or "nothing yet"
        raise ToolError(f"look has not asked that; it has asked {known}")
    return asked


def _meets(judged: Judged | None, want: str) -> bool | None:
    if judged is None or judged.verdict not in ("yes", "no"):
        return None
    return judged.verdict == want


# None is not known: the answers that are known do not settle it.
def _together(states: Sequence[bool | None], how: str) -> bool | None:
    settled, opposite = (False, True) if how == "all" else (True, False)
    if settled in states:
        return settled
    return None if None in states else opposite


# A wording that holds something already found means that thing: "box on the belt" after "box".
# Tracking it under the new words would take minutes and find the same items.
def _known_thing(bench: Bench, text: str) -> str:
    thing = _thing(text)
    asked = f" {_stems(thing)} "
    for known in (*bench.found, *bench.session.things):
        if f" {_stems(known)} " in asked:
            return known
    return thing


def _stems(thing: str) -> str:
    return " ".join(word.rstrip("s") for word in thing.split())


def _wait_words(seconds: float) -> str:
    return "under a minute" if seconds < 60 else f"about {round(seconds / 60)} min"


# Another moment that shows the item well, as far as possible from the view already used.
def _second_view(item: Item) -> Observation | None:
    best = _size(item.best.box)
    views = [o for o in item.track.observations if _size(o.box) >= SECOND_VIEW_SIZE * best]
    view = max(views, key=lambda o: abs(o.t - item.best.t), default=None)
    return None if view is None or view.t == item.best.t else view


def _size(box: Box) -> float:
    return (box.x1 - box.x0) * (box.y1 - box.y0)


# What is written on the box in a proof; a bare yes means nothing without its question.
def _tag(answer: _TileAnswer, question: str) -> str:
    bare = answer.answer.lower().rstrip(".") in ("yes", "no", "cannot tell")
    return question if bare else answer.answer


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


def _middle_x(box: Box) -> float:
    return (box.x0 + box.x1) / 2


def _route(first: Box, last: Box) -> str:
    if max(abs(_middle_x(first) - _middle_x(last)), abs(_middle_y(first) - _middle_y(last))) < 0.1:
        return f"it stays at the {_region(first)} of the frame"
    return f"it goes from the {_region(first)} of the frame to the {_region(last)}"


def _region(box: Box) -> str:
    row = ("top", "", "bottom")[min(2, int(_middle_y(box) * 3))]
    column = ("left", "", "right")[min(2, int(_middle_x(box) * 3))]
    return f"{row} {column}".strip() or "middle"


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


def _spans(bench: Bench) -> tuple[TimeSpan, ...]:
    whole = TimeSpan(start=0, end=bench.footage.duration_s)
    return tuple(bench.session.active_spans) or (whole,)


# Moments spread evenly over these stretches, inside them, away from their ends.
def _moments(spans: Sequence[TimeSpan]) -> tuple[float, ...]:
    total = sum(_length(span) for span in spans)
    moments: list[float] = []
    for n in range(NAME_MOMENTS):
        left = total * (2 * n + 1) / (2 * NAME_MOMENTS)
        for span in spans:
            if left <= _length(span):
                moments.append(span.start + left)
                break
            left -= _length(span)
    return tuple(moments)


# A moment filling most of the video says nothing about when: it is a general impression, so it
# gives the agent no times to repeat and no proof to show.
def note_moment(
    notebook: Notebook, moment: Moment, duration_s: float, clip: TimeSpan | None
) -> Evidence:
    if moment.span.end - moment.span.start >= BROAD_SHARE * duration_s:
        return notebook.add("seen", f"Throughout: {moment.what}")
    return notebook.add("seen", f"{_span_text(moment.span)}: {moment.what}", clip)


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
