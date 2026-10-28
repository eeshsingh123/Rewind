from collections.abc import Sequence
from dataclasses import dataclass

from rewind.contracts import Checkpoint, EndOfFootage, EntersZone, ExitsView, Rule, Violation
from rewind.core.analysis.proof import short_reason

EXAMPLES = 3


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


# "Rule 1 · Every box needs a label.\n❌ 2 of 9 boxes failed: the boxes at 00:04 and 00:14."
def rule_summary(outcome: RuleOutcome) -> str:
    entity = outcome.rule.spec.entity.phrase
    title = f"Rule {outcome.number} · {outcome.rule.text}"
    items = [v for v in outcome.violations if v.track_id is not None]
    line_wide = [v for v in outcome.violations if v.track_id is None]
    failed = len({v.track_id for v in items})
    judged = max(outcome.judged, failed)
    lines = [title]
    # Only items seen reaching the checkpoint are judged, so the count says which ones they are.
    which = _reached(outcome.rule.spec.checkpoint)
    if failed:
        times = sorted(set(map(clock_time, (v.t for v in items))))
        lead = ", for example" if len(times) > EXAMPLES else ":"
        shown = plural(entity, min(failed, EXAMPLES))
        lines.append(
            f"❌ {failed} of {judged} {plural(entity, judged)}{which} failed{lead} "
            f"the {shown} at {_join(times[:EXAMPLES])}."
        )
    elif judged:
        lines.append(f"✅ All {judged} {plural(entity, judged)}{which} passed.")
    elif not line_wide:
        lines.append(f"No {entity} reached the point where this rule checks it.")
    lines += [f"❌ {clock_time(v.t)}: {v.reason}." for v in line_wide[:EXAMPLES]]
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


def _reached(checkpoint: Checkpoint) -> str:
    match checkpoint:
        case ExitsView():
            return " that left the view"
        case EntersZone(zone=zone):
            return f" that entered {zone.replace('_', ' ')}"
        case EndOfFootage():
            return ""


def _join(words: Sequence[str]) -> str:
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]
