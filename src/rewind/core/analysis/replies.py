from collections.abc import Sequence
from dataclasses import dataclass

from rewind.contracts import (
    Checkpoint,
    CountAtLeast,
    EndOfFootage,
    EntersZone,
    ExitsView,
    Has,
    InZone,
    Rule,
    RuleSpec,
    Violation,
)
from rewind.core.analysis.proof import short_reason

EXAMPLES = 3
# When nearly every item lacks the part, the likelier story is that the part's name does not
# match how it looks on camera, so the result says so rather than reporting a broken line.
SUSPECT_SHARE = 0.8
SUSPECT_FROM = 5


@dataclass(frozen=True, slots=True)
class RuleOutcome:
    number: int
    rule: Rule
    # Items the rule judged; items still in view at the end, or never seen whole, are not.
    judged: int
    violations: tuple[Violation, ...]


def clock_time(t: float) -> str:
    minutes, seconds = divmod(int(t), 60)
    return f"{minutes:02d}:{seconds:02d}"


# "0:42", "12:30": how long a video is, the way a video player shows it.
def video_length(seconds: float) -> str:
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes}:{rest:02d}"


def duration_text(seconds: float) -> str:
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes} min {rest} s" if minutes else f"{rest} s"


def plural(phrase: str, count: int) -> str:
    if count == 1:
        return phrase
    if phrase.endswith(("s", "x", "z", "ch", "sh")):
        return phrase + "es"
    if phrase.endswith("y") and phrase[-2:-1] not in ("a", "e", "i", "o", "u"):
        return phrase[:-1] + "ies"
    return phrase + "s"


def concept_of(violation: Violation) -> str | None:
    return None if violation.track_id is None else violation.track_id.rpartition("#")[0]


# A rule as one plain sentence: "every box has a label when it leaves the view".
def rule_sentence(spec: RuleSpec) -> str:
    entity = spec.entity.phrase

    def part(key: str) -> str:
        return spec.parts.get(key, key.replace("_", " "))

    clauses: list[str] = []
    for requirement in spec.requirements:
        match requirement.predicate:
            case Has(part=key):
                text = f"has {_article(part(key))} {part(key)}"
            case CountAtLeast(part=key, n=n):
                text = f"has at least {n} {plural(part(key), n)}"
            case InZone(zone=zone):
                text = f"is in the {_zone(zone)}"
        if requirement.timing == "ever_before":
            text += " at some point"
        clauses.append(text)
    clauses += [f"gets its {part(step.first)} before its {part(step.then)}" for step in spec.order]
    sentences: list[str] = []
    if clauses:
        sentences.append(f"every {entity} {join_words(clauses)} {_when(spec.checkpoint)}")
    if spec.dwell is not None:
        sentences.append(
            f"no {entity} stays in the {_zone(spec.dwell.zone)} longer than {spec.dwell.max_s:g} s"
        )
    if spec.rate is not None:
        sentences.append(
            f"at least {spec.rate.min_per_minute:g} {plural(entity, 2)} pass every minute"
        )
    return "; ".join(sentences)


# One block per rule for the result message:
# "❌ *2 of 9 boxes left the view without a label*\nThe box at 00:14 is the first one (in red).
# \nRule 1 · every box has a label when it leaves the view."
def rule_result(outcome: RuleOutcome, clipped: Violation | None) -> str:
    spec = outcome.rule.spec
    entity = spec.entity.phrase
    items = [v for v in outcome.violations if v.track_id is not None]
    line_wide = [v for v in outcome.violations if v.track_id is None]
    failed = len({v.track_id for v in items})
    judged = max(outcome.judged, failed)
    lines: list[str] = []
    moment = _moment(spec.checkpoint)
    needed = _single_part(spec)
    if failed:
        what = f"{moment} without {_article(needed)} {needed}" if needed else "broke the rule"
        what = what.strip()
        lines.append(f"❌ *{failed} of {judged} {plural(entity, judged)} {what}*")
        times = sorted(set(map(clock_time, (v.t for v in items))))
        if clipped is not None and clipped.track_id is not None:
            first = clock_time(clipped.t)
            lines.append(f"The {entity} at {first} is the one in the clip (in red).")
            times = [t for t in times if t != first]
            if times:
                more = ", for example" if len(times) > EXAMPLES else ""
                lines.append(f"Others{more}: {join_words(times[:EXAMPLES])}.")
        else:
            lead = ", for example" if len(times) > EXAMPLES else ""
            lines.append(f"At{lead} {join_words(times[:EXAMPLES])}.")
        if needed and judged >= SUSPECT_FROM and failed >= SUSPECT_SHARE * judged:
            lines.append(
                f"Nearly every {entity} failed, so look at the clip first: if they do have "
                f"{_article(needed)} {needed}, I may not recognise it by that name. Try saying "
                "how it looks, such as its colour or what is printed on it."
            )
    elif judged:
        that = f" that {moment}" if moment else ""
        had = f"had {_article(needed)} {needed}" if needed else "passed"
        lines.append(f"✅ *All {judged} {plural(entity, judged)}{that} {had}.*")
    elif not line_wide:
        lines.append(f"No {entity} reached the point where this rule checks it.")
    lines += [f"❌ {clock_time(v.t)}: {v.reason}." for v in line_wide[:EXAMPLES]]
    lines.append(f"Rule {outcome.number} · {rule_sentence(spec)}.")
    return "\n".join(lines)


def clip_caption(violation: Violation, number: int, count: int, rule_number: int | None) -> str:
    rule = f"Rule {rule_number}" if rule_number else "A removed rule"
    concept = concept_of(violation)
    what = f"The {concept} at {clock_time(violation.t)}" if concept else clock_time(violation.t)
    return f"{rule} · clip {number} of {count}\n{what}: {short_reason(violation.reason)}."


# Facts for the agent's own tools: one line per finding.
def violation_line(violation: Violation, rule_number: int | None) -> str:
    concept = concept_of(violation) or "line"
    rule = f"rule {rule_number}" if rule_number else "a removed rule"
    return f"{clock_time(violation.t)} · the {concept}: {violation.reason} ({rule})"


# The part a rule asks for, when it asks for exactly one at its checkpoint; it names the failure.
def _single_part(spec: RuleSpec) -> str | None:
    if spec.order or len(spec.requirements) != 1:
        return None
    match spec.requirements[0].predicate:
        case Has(part=key):
            return spec.parts.get(key, key)
        case _:
            return None


def _moment(checkpoint: Checkpoint) -> str:
    match checkpoint:
        case ExitsView():
            return "left the view"
        case EntersZone(zone=zone):
            return f"entered the {_zone(zone)}"
        case EndOfFootage():
            return ""


def _when(checkpoint: Checkpoint) -> str:
    match checkpoint:
        case ExitsView():
            return "when it leaves the view"
        case EntersZone(zone=zone):
            return f"when it enters the {_zone(zone)}"
        case EndOfFootage():
            return "by the end of the video"


def _zone(key: str) -> str:
    return key.replace("_", " ")


def _article(phrase: str | None) -> str:
    return "an" if (phrase or "")[:1].lower() in ("a", "e", "i", "o", "u") else "a"


def join_words(words: Sequence[str]) -> str:
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]
