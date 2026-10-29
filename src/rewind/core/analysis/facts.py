import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rewind.contracts import Box, TimeSpan, Violation
from rewind.core.analysis.patterns import NOTABLE_P, Pattern, compare, patterns
from rewind.core.analysis.replies import clock_time, duration_text, join_words, plural
from rewind.core.analysis.timeline import Judged, Stop, Timeline, usual_gap

# A clip longer than this would be too big for a chat; a longer stretch shows its start.
MAX_CLIP_S = 20.0
CLIP_LEAD_S = 2.0
# Failures listed one by one; past this the rule's summary still counts them all.
MAX_FAILURES_LISTED = 20
MAX_MINUTES_LISTED = 30
# Per-minute counts help only when there is more than a minute or two to compare.
MINUTES_FROM_S = 120.0


@dataclass(frozen=True, slots=True)
class Tile:
    # A moment of the footage shown to the model, with the item drawn, labelled by fact id.
    t: float
    box: Box | None
    label: str


@dataclass(frozen=True, slots=True)
class Fact:
    id: str
    text: str
    clip: TimeSpan | None = None
    violation_ids: tuple[str, ...] = ()
    tile: Tile | None = None
    # A stop, a pattern chance would rarely make, or a real change from earlier videos.
    notable: bool = False


@dataclass(frozen=True, slots=True)
class RuleView:
    rule_id: str
    number: int
    sentence: str
    entity: str


@dataclass(frozen=True, slots=True)
class Earlier:
    # How many other recordings of the line this rule was checked on, and what it judged there.
    recordings: int
    judged: tuple[Judged, ...]


@dataclass(frozen=True, slots=True)
class SheetInput:
    name: str
    timeline: Timeline
    rules: tuple[RuleView, ...]
    violations: tuple[Violation, ...]
    earlier: Mapping[str, Earlier]
    # Where each failed item was last seen whole, for its picture.
    boxes: Mapping[str, tuple[float, Box]]
    window: TimeSpan | None = None


def fact_sheet(given: SheetInput) -> tuple[Fact, ...]:
    facts: list[tuple[str, TimeSpan | None, tuple[str, ...], Tile | None, bool]] = []
    timeline = given.timeline
    by_id = {v.id: v for v in given.violations}

    def add(text: str, clip: TimeSpan | None = None, ids: Sequence[str] = (),
            tile: Tile | None = None, notable: bool = False) -> None:  # fmt: skip
        facts.append((text, clip, tuple(ids), tile, notable))

    add(f"The recording {given.name} is {duration_text(timeline.duration_s)} long. "
        "Times are minutes:seconds from its start.")  # fmt: skip
    for stop in timeline.stops:
        add(_stop_text(stop, _entity(given.rules)), _stop_clip(stop, timeline.duration_s),
            tile=Tile(_stop_moment(stop), None, ""), notable=True)  # fmt: skip
    pace = _pace(timeline, given.rules)
    if pace:
        add(pace)
    for rule in given.rules:
        judged = timeline.of_rule(rule.rule_id)
        failed = [by_id[j.violation_id] for j in judged if j.violation_id in by_id]
        line_wide = [v for v in given.violations if v.rule_id == rule.rule_id
                     and v.track_id is None]  # fmt: skip
        add(_summary(rule, judged, failed), failed[0].clip if failed else None,
            [v.id for v in failed])  # fmt: skip
        for violation in sorted(failed, key=lambda v: v.t)[:MAX_FAILURES_LISTED]:
            seen = given.boxes.get(violation.id)
            tile = Tile(seen[0], seen[1], "") if seen else None
            add(f"The {rule.entity} at {clock_time(violation.t)} broke rule {rule.number}: "
                f"{violation.reason}.", violation.clip, [violation.id], tile)  # fmt: skip
        for violation in line_wide[:MAX_FAILURES_LISTED]:
            add(f"At {clock_time(violation.t)} the line broke rule {rule.number}: "
                f"{violation.reason}.", violation.clip, [violation.id])  # fmt: skip
        found = patterns(judged, timeline.stops)
        for pattern in found:
            if pattern.notable:
                ids = [j.violation_id for j in pattern.items if j.violation_id]
                add(_pattern_text(rule, pattern), _span_clip(ids, by_id), ids, notable=True)
        if len(failed) >= 3 and not any(p.notable for p in found):
            add(f"Rule {rule.number}: no pattern stands out in when its failures happened; "
                "they look spread out.")  # fmt: skip
        earlier = given.earlier.get(rule.rule_id)
        if earlier is not None and earlier.judged:
            text, notable = _earlier_text(rule, judged, earlier)
            add(text, failed[0].clip if failed and notable else None, notable=notable)
        if timeline.duration_s >= MINUTES_FROM_S:
            for text in _minutes(rule, judged, timeline.duration_s):
                add(text)
        if given.window is not None:
            add(_window_text(rule, judged, given.window))
    numbered: list[Fact] = []
    for index, (text, clip, ids, tile, notable) in enumerate(facts, 1):
        fact_id = f"F{index}"
        if tile is not None:
            tile = Tile(tile.t, tile.box, f"{fact_id} {clock_time(tile.t)}")
        numbered.append(Fact(fact_id, text, clip, ids, tile, notable))
    return tuple(numbered)


def sheet_text(facts: Sequence[Fact]) -> str:
    lines: list[str] = []
    for fact in facts:
        marks = [m for m, on in (("notable", fact.notable), ("has a clip", fact.clip is not None),
                                 ("in the picture", fact.tile is not None)) if on]  # fmt: skip
        lines.append(f"{fact.id}: {fact.text}" + (f" [{', '.join(marks)}]" if marks else ""))
    return "\n".join(lines)


# The moments shown in the picture: stops first, then failures that are part of a pattern.
def picture_tiles(facts: Sequence[Fact], limit: int) -> tuple[Tile, ...]:
    in_patterns = {vid for f in facts if f.notable for vid in f.violation_ids}

    def rank(fact: Fact) -> tuple[int, float]:
        assert fact.tile is not None
        first = 0 if fact.tile.box is None else 1 if set(fact.violation_ids) & in_patterns else 2
        return first, fact.tile.t

    tiled = sorted((f for f in facts if f.tile is not None), key=rank)
    return tuple(f.tile for f in tiled[:limit] if f.tile is not None)


_NUMBER = re.compile(r"\d+(?:[:.]\d+)*")
_WORDS = {
    "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
}  # fmt: skip
_WORD = re.compile(r"\b(" + "|".join(_WORDS) + r")\b", re.IGNORECASE)


# Numbers and times a sentence states that none of its facts state: a claim code cannot trace.
def untraceable(text: str, facts: Sequence[Fact]) -> tuple[str, ...]:
    known = {_canonical(n) for fact in facts for n in _NUMBER.findall(fact.text)}
    spoken = _WORD.sub(lambda m: _WORDS[m.group(1).lower()], text)
    stated = [n for n in _NUMBER.findall(spoken) if _canonical(n) not in {"0", "1"}]
    return tuple(dict.fromkeys(n for n in stated if _canonical(n) not in known))


def _canonical(number: str) -> str:
    if ":" in number:
        seconds = 0
        for part in number.split(":"):
            if not part.isdigit():
                return number
            seconds = seconds * 60 + int(part)
        return f"t{seconds}"
    try:
        return f"{float(number):g}"
    except ValueError:
        return number


def _entity(rules: Sequence[RuleView]) -> str:
    entities = {rule.entity for rule in rules}
    return next(iter(entities)) if len(entities) == 1 else "item"


def _summary(rule: RuleView, judged: Sequence[Judged], failed: Sequence[Violation]) -> str:
    head = f"Rule {rule.number} ({rule.sentence})"
    if not judged:
        return f"{head}: no {rule.entity} reached the point where this rule checks it."
    times = join_words(sorted({clock_time(v.t) for v in failed})[:MAX_FAILURES_LISTED])
    if not failed:
        return f"{head}: all {len(judged)} {plural(rule.entity, len(judged))} judged passed."
    return (
        f"{head}: {len(judged)} {plural(rule.entity, len(judged))} judged, "
        f"{len(failed)} failed, at {times}."
    )


def _pace(timeline: Timeline, rules: Sequence[RuleView]) -> str | None:
    ids = timeline.rule_ids()
    if not ids:
        return None
    busiest = max(ids, key=lambda rule_id: len(timeline.of_rule(rule_id)))
    usual = usual_gap([j.t for j in timeline.of_rule(busiest)])
    if usual is None:
        return None
    # From the usual gap, not a count: an item the camera check missed would lower a count.
    per_minute = round(60 / usual)
    entity = next((r.entity for r in rules if r.rule_id == busiest), "item")
    return (
        f"While the line was moving, one {entity} came through about every {usual:.1f} s, "
        f"about {per_minute} {plural(entity, per_minute)} a minute."
    )


def _stop_text(stop: Stop, entity: str) -> str:
    stood = stop.stood
    seen = (
        "nothing moved in view"
        if stop.seen == "no_motion"
        else f"the {plural(entity, 2)} in view stood still"
    )
    return (
        f"Stop: from {clock_time(stood.start)} to {clock_time(stood.end)} "
        f"({duration_text(stood.end - stood.start)}) {seen}, so the line was stopped. "
        f"Usually one {entity} comes through every {stop.usual_gap_s:.1f} s, so about "
        f"{stop.lost_items} {plural(entity, stop.lost_items)} of output were lost."
    )


def _stop_moment(stop: Stop) -> float:
    return (stop.stood.start + stop.stood.end) / 2


def _stop_clip(stop: Stop, duration_s: float) -> TimeSpan:
    span = stop.stood
    start = max(0.0, span.start - CLIP_LEAD_S)
    return TimeSpan(start=start, end=min(duration_s, span.end + CLIP_LEAD_S, start + MAX_CLIP_S))


def _span_clip(ids: Sequence[str], by_id: Mapping[str, Violation]) -> TimeSpan | None:
    clips = [by_id[i].clip for i in ids if i in by_id]
    if not clips:
        return None
    start = min(c.start for c in clips)
    end = max(c.end for c in clips)
    if end - start > MAX_CLIP_S:
        return clips[0]
    return TimeSpan(start=start, end=end)


def _pattern_text(rule: RuleView, pattern: Pattern) -> str:
    entities = plural(rule.entity, 2)
    times = join_words([clock_time(j.t) for j in pattern.items])
    head = f"Pattern in rule {rule.number}"
    chance = _chance(pattern.p)
    match pattern.kind:
        case "in_a_row":
            body = f"{pattern.numbers[0]} {entities} in a row failed, at {times}"
        case "every_nth":
            period, *places = pattern.numbers
            body = (
                f"the failures fall on every {_ordinal(period)} {rule.entity} "
                f"(the {join_words([_ordinal(p) for p in places])} in order), at {times}; "
                "a fault on every Nth item is a sign of one faulty head, nozzle or lane "
                "on a machine that serves items in turn"
            )
        case "after_restart":
            hits, size, other, other_size = pattern.numbers
            body = (
                f"{hits} of the {size} {entities} right after the line restarted failed "
                f"(at {times}), against {other} of the other {other_size}"
            )
        case "started":
            before, before_n, after, after_n = pattern.numbers
            body = (
                f"failures started at {clock_time(pattern.at or 0)}: {before} of {before_n} "
                f"failed before it, {after} of {after_n} from then on"
            )
        case _:
            before, before_n, after, after_n = pattern.numbers
            body = (
                f"failures stopped at {clock_time(pattern.at or 0)}: {before} of {before_n} "
                f"failed before it, {after} of {after_n} from then on"
            )
    return f"{head}: {body}. {chance}"


def _earlier_text(rule: RuleView, judged: Sequence[Judged], earlier: Earlier) -> tuple[str, bool]:
    old_failed = sum(not j.passed for j in earlier.judged)
    now_failed = sum(not j.passed for j in judged)
    videos = plural("recording", earlier.recordings)
    text = (
        f"Rule {rule.number} on {earlier.recordings} other {videos} of this line: "
        f"{old_failed} of {len(earlier.judged)} {plural(rule.entity, len(earlier.judged))} "
        f"failed. In this recording: {now_failed} of {len(judged)}."
    )
    p = compare(judged, earlier.judged)
    if p < NOTABLE_P:
        return f"{text} That is a real change. {_chance(p)}", True
    return f"{text} That is about the same as before.", False


def _minutes(rule: RuleView, judged: Sequence[Judged], duration_s: float) -> list[str]:
    texts: list[str] = []
    for minute in range(min(MAX_MINUTES_LISTED, math.ceil(duration_s / 60))):
        inside = [j for j in judged if minute * 60 <= j.t < (minute + 1) * 60]
        failed = sum(not j.passed for j in inside)
        texts.append(
            f"Rule {rule.number}, {clock_time(minute * 60)} to {clock_time((minute + 1) * 60)}: "
            f"{len(inside)} {plural(rule.entity, len(inside))} judged, {failed} failed."
        )
    return texts


def _window_text(rule: RuleView, judged: Sequence[Judged], window: TimeSpan) -> str:
    inside = [j for j in judged if window.start <= j.t <= window.end]
    failed = [j for j in inside if not j.passed]
    text = (
        f"Rule {rule.number}, from {clock_time(window.start)} to {clock_time(window.end)}: "
        f"{len(inside)} {plural(rule.entity, len(inside))} judged, {len(failed)} failed"
    )
    if failed:
        text += f", at {join_words([clock_time(j.t) for j in failed])}"
    return text + "."


def _chance(p: float) -> str:
    if p <= 0:
        return "Chance alone would almost never do this."
    odds = 1 / p
    rounded = round(odds, -math.floor(math.log10(odds))) if odds >= 10 else round(odds)
    return f"Chance alone would do this about 1 time in {rounded:g}."


def _ordinal(number: int) -> str:
    suffix = (
        "th" if 10 <= number % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    )
    return f"{number}{suffix}"
