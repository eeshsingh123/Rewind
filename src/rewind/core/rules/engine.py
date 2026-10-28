import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

from rewind.contracts import (
    Checkpoint,
    Dwell,
    EndOfFootage,
    EntersZone,
    ExitsView,
    InZone,
    Observation,
    Polygon,
    Rate,
    Requirement,
    RuleSpec,
    TimeSpan,
    Track,
    Tripwire,
    ViewExit,
    Violation,
    Zone,
)
from rewind.core.rules.items import Item, build_items
from rewind.core.rules.predicates import PREDICATES, Evaluator
from rewind.core.rules.zones import VIEW_EXIT, RuleError, area, center, contains, crossing

# `at_checkpoint` looks back this far from the checkpoint; perception samples parts in this window.
CHECKPOINT_WINDOW_S = 1.0
# A track that lasts until this close to the end of the footage has not left the view.
EXIT_MARGIN_S = 1.0
CLIP_BEFORE_S = 4.0
CLIP_AFTER_S = 2.0
RATE_WINDOW_S = 60.0


@dataclass(frozen=True, slots=True)
class Scene:
    footage_id: str
    duration_s: float
    # Every track and observation of the footage; the engine picks the rule's concepts.
    tracks: tuple[Track, ...]
    observations: tuple[Observation, ...]
    zones: tuple[Zone, ...] = ()


def evaluate(
    rule_id: str,
    spec: RuleSpec,
    scene: Scene,
    predicates: Mapping[str, Evaluator] = PREDICATES,
) -> tuple[Violation, ...]:
    zones = _zones(spec, scene)
    items = build_items(spec, scene.tracks, scene.observations, zones)
    found: list[Violation] = []
    passed: list[float] = []
    for item in items:
        t = checkpoint_time(spec.checkpoint, item, scene.duration_s)
        if t is not None:
            passed.append(t)
            failures = _requirement_failures(spec, item, t, predicates) + _order_failures(
                spec, item, t
            )
            if failures:
                found.append(_item_violation(rule_id, scene, item, t, "; ".join(failures)))
        if spec.dwell is not None:
            found.extend(_dwell_violations(rule_id, scene, spec.dwell, item))
    if spec.rate is not None:
        found.extend(_rate_violations(rule_id, scene, spec, spec.rate, passed))
    return tuple(sorted(found, key=lambda violation: (violation.t, violation.id)))


def zone_keys(spec: RuleSpec) -> frozenset[str]:
    keys = {r.predicate.zone for r in spec.requirements if isinstance(r.predicate, InZone)}
    if isinstance(spec.checkpoint, EntersZone):
        keys.add(spec.checkpoint.zone)
    if spec.dwell is not None:
        keys.add(spec.dwell.zone)
    return frozenset(keys)


def checkpoint_time(checkpoint: Checkpoint, item: Item, duration_s: float) -> float | None:
    match checkpoint:
        case ExitsView():
            return _exit_time(item, duration_s)
        case EndOfFootage():
            # The item's last known state stands for its state at the end of the footage.
            return item.end
        case EntersZone(zone=key):
            return _entry_time(item, item.zones[key], duration_s)


def _zones(spec: RuleSpec, scene: Scene) -> dict[str, Zone]:
    zones = {VIEW_EXIT.key: VIEW_EXIT} | {zone.key: zone for zone in scene.zones}
    missing = sorted(zone_keys(spec) - zones.keys())
    if missing:
        raise RuleError(f"zones not defined for this line: {', '.join(missing)}")
    return zones


def _exit_time(item: Item, duration_s: float) -> float | None:
    if item.end >= duration_s - EXIT_MARGIN_S:
        return None
    return item.end


def _entry_time(item: Item, zone: Zone, duration_s: float) -> float | None:
    match zone.shape:
        case ViewExit():
            return _exit_time(item, duration_s)
        case Polygon() as polygon:
            return next((o.t for o in item.path if contains(polygon, center(o.box))), None)
        case Tripwire() as tripwire:
            return _crossing_time(item, tripwire)


def _crossing_time(item: Item, tripwire: Tripwire) -> float | None:
    for before, after in zip(item.path, item.path[1:], strict=False):
        share = crossing(tripwire, center(before.box), center(after.box))
        if share is not None:
            return before.t + share * (after.t - before.t)
    return None


def _requirement_failures(
    spec: RuleSpec, item: Item, t: float, predicates: Mapping[str, Evaluator]
) -> list[str]:
    failures: list[str] = []
    for requirement in spec.requirements:
        evaluator = predicates.get(requirement.predicate.kind)
        if evaluator is None:
            raise RuleError(f"no evaluator for predicate {requirement.predicate.kind}")
        failure = evaluator(requirement.predicate, item, _requirement_span(requirement, item, t))
        if failure is not None:
            failures.append(f"{failure} {_moment(spec.checkpoint, requirement)}")
    return failures


def _requirement_span(requirement: Requirement, item: Item, t: float) -> TimeSpan:
    if requirement.timing == "ever_before":
        return TimeSpan(start=item.start, end=t)
    return TimeSpan(start=max(item.start, t - CHECKPOINT_WINDOW_S), end=t)


def _moment(checkpoint: Checkpoint, requirement: Requirement) -> str:
    match checkpoint:
        case ExitsView():
            event = "it left the view"
        case EndOfFootage():
            event = "the footage ended"
        case EntersZone(zone=key):
            event = f"it entered {key}"
    return f"{'before' if requirement.timing == 'ever_before' else 'when'} {event}"


def _order_failures(spec: RuleSpec, item: Item, t: float) -> list[str]:
    span = TimeSpan(start=item.start, end=t)
    failures: list[str] = []
    for step in spec.order:
        then_seen = item.parts_in(step.then, span)
        if not then_seen:
            continue
        first_seen = item.parts_in(step.first, span)
        # Seen in the same frame counts as in order: sampling cannot tell them apart.
        if not first_seen or first_seen[0].t > then_seen[0].t:
            first, then = item.part_phrases[step.first], item.part_phrases[step.then]
            failures.append(f"{then} came before {first}")
    return failures


def _dwell_violations(rule_id: str, scene: Scene, dwell: Dwell, item: Item) -> list[Violation]:
    zone = area(item.zones, dwell.zone)
    found: list[Violation] = []
    for stay in _stays(item, zone):
        if stay.end - stay.start <= dwell.max_s:
            continue
        t = stay.start + dwell.max_s
        found.append(
            Violation(
                id=_violation_id(
                    rule_id, scene.footage_id, item.track.track_id, f"dwell@{stay.start:.3f}"
                ),
                rule_id=rule_id,
                footage_id=scene.footage_id,
                track_id=item.track.track_id,
                t=t,
                reason=(
                    f"stayed in {dwell.zone} for {stay.end - stay.start:.1f} s, "
                    f"more than {dwell.max_s:g} s"
                ),
                clip=TimeSpan(start=stay.start, end=min(scene.duration_s, t + CLIP_AFTER_S)),
            )
        )
    return found


def _stays(item: Item, zone: Polygon) -> list[TimeSpan]:
    stays: list[TimeSpan] = []
    start: float | None = None
    last = 0.0
    for observation in item.path:
        if contains(zone, center(observation.box)):
            start = observation.t if start is None else start
            last = observation.t
        elif start is not None:
            stays.append(TimeSpan(start=start, end=last))
            start = None
    if start is not None:
        stays.append(TimeSpan(start=start, end=last))
    return stays


def _rate_violations(
    rule_id: str, scene: Scene, spec: RuleSpec, rate: Rate, passed: list[float]
) -> list[Violation]:
    found: list[Violation] = []
    for window in _rate_windows(scene.duration_s):
        count = sum(window.start <= t < window.end for t in passed)
        needed = rate.min_per_minute * (window.end - window.start) / RATE_WINDOW_S
        if count >= needed:
            continue
        found.append(
            Violation(
                id=_violation_id(rule_id, scene.footage_id, "line", f"rate@{window.start:.3f}"),
                rule_id=rule_id,
                footage_id=scene.footage_id,
                track_id=None,
                t=window.start,
                reason=(
                    f"{count} {spec.entity.phrase} passed between {window.start:g} s and "
                    f"{window.end:g} s, fewer than {needed:g}"
                ),
                clip=window,
            )
        )
    return found


def _rate_windows(duration_s: float) -> list[TimeSpan]:
    # Whole minutes only, so a short tail is not judged on a handful of items; shorter footage
    # is judged as one window, pro rata.
    if duration_s < RATE_WINDOW_S:
        return [TimeSpan(start=0, end=duration_s)]
    minutes = int(duration_s // RATE_WINDOW_S)
    return [TimeSpan(start=m * RATE_WINDOW_S, end=(m + 1) * RATE_WINDOW_S) for m in range(minutes)]


def _item_violation(rule_id: str, scene: Scene, item: Item, t: float, reason: str) -> Violation:
    return Violation(
        id=_violation_id(rule_id, scene.footage_id, item.track.track_id, "checkpoint"),
        rule_id=rule_id,
        footage_id=scene.footage_id,
        track_id=item.track.track_id,
        t=t,
        reason=reason,
        clip=TimeSpan(
            start=max(0.0, t - CLIP_BEFORE_S), end=max(t, min(scene.duration_s, t + CLIP_AFTER_S))
        ),
    )


# Derived from what was found, not from a counter, so re-running a rule upserts the same rows.
def _violation_id(rule_id: str, footage_id: str, subject: str, check: str) -> str:
    digest = hashlib.sha256("\x1f".join((rule_id, footage_id, subject, check)).encode())
    return digest.hexdigest()[:32]
