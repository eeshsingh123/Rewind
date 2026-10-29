import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from rewind.contracts import (
    Box,
    Observation,
    Point,
    Polygon,
    RuleSpec,
    TimeSpan,
    Track,
    Tripwire,
    Zone,
)
from rewind.core.ports import ConceptPrompt
from rewind.core.rules import (
    PREDICATES,
    Check,
    Keyframe,
    RuleError,
    Scene,
    checks,
    evaluate,
    find_zone,
)
from tests.fakes import FakeSegmenter

FOOTAGE = "f1"


def at(concept: str, t: float, x: float, y: float, size: float = 0.1) -> Observation:
    half = size / 2
    box = Box(x0=x - half, y0=y - half, x1=x + half, y1=y + half)
    return Observation(footage_id=FOOTAGE, t=t, concept=concept, box=box, score=0.9)


def track(
    track_id: str, points: Sequence[tuple[float, float]], y: float = 0.5, concept: str = "jar"
):
    observations = tuple(at(concept, t, x, y, size=0.2) for t, x in points)
    return Track(footage_id=FOOTAGE, track_id=track_id, concept=concept, observations=observations)


# Crosses the frame left to right in 8 s: x = 0.1 + 0.1 * (t - start), sampled at 2 fps.
def crossing(track_id: str, start: float = 0.0, y: float = 0.5) -> Track:
    return track(track_id, [(start + i / 2, 0.1 + i / 20) for i in range(17)], y)


# A part seen on the item at time t (the item is sampled at that time).
def on(item: Track, concept: str, t: float, dx: float = 0.0) -> Observation:
    seen = next(o for o in item.observations if o.t == t)
    return at(
        concept, t, (seen.box.x0 + seen.box.x1) / 2 + dx, (seen.box.y0 + seen.box.y1) / 2, 0.04
    )


def rule(**fields: Any) -> RuleSpec:
    base: dict[str, Any] = {
        "entity": {"key": "jar", "phrase": "jar"},
        "checkpoint": {"kind": "exits_view"},
    }
    return RuleSpec.model_validate(base | fields)


def needs(kind: str, timing: str = "at_checkpoint", **fields: Any) -> dict[str, Any]:
    return {"predicate": {"kind": kind, **fields}, "timing": timing}


def area(key: str, x0: float, y0: float, x1: float, y1: float) -> Zone:
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    shape = Polygon(points=tuple(Point(x=x, y=y) for x, y in corners))
    return Zone(key=key, shape=shape, source="drawn")


def scene(
    tracks: list[Track],
    parts: list[Observation],
    zones: tuple[Zone, ...] = (),
    duration: float = 30,
):
    return Scene(FOOTAGE, duration, tuple(tracks), tuple(parts), zones)


def test_at_checkpoint_needs_the_part_just_before_the_item_leaves_the_view() -> None:
    a, b = crossing("a", y=0.3), crossing("b", y=0.7)
    still_in_view = crossing("c", start=22)
    parts = [on(a, "cap", 7.5), on(b, "cap", 2.0)]
    spec = rule(parts={"cap": "cap"}, requirements=[needs("has", part="cap")])

    [violation] = evaluate("r1", spec, scene([a, b, still_in_view], parts))

    assert (violation.track_id, violation.t) == ("b", 8.0)
    assert violation.reason == "no cap when it left the view"
    assert violation.clip == TimeSpan(start=5.0, end=9.0)


# Whole in the frame until 5.5 s, then a sliver at the right border for 2 s before it is gone.
def slow_exit(track_id: str, y: float = 0.5) -> Track:
    whole = [at("jar", t / 2, 0.3 + t / 20, y, 0.2) for t in range(12)]
    sliver = [
        Observation(
            footage_id=FOOTAGE, t=6.0 + i / 2, concept="jar",
            box=Box(x0=0.85 + i / 20, y0=y - 0.1, x1=1.0, y1=y + 0.1), score=0.9,
        )
        for i in range(4)
    ]  # fmt: skip
    return Track(
        footage_id=FOOTAGE, track_id=track_id, concept="jar", observations=(*whole, *sliver)
    )


def test_an_exit_is_checked_while_the_item_is_still_whole_in_the_frame() -> None:
    labelled, bare = slow_exit("labelled", y=0.3), slow_exit("bare", y=0.7)
    # The label is cut off by the border in the sliver, as on a real belt.
    parts = [on(labelled, "label", t) for t in (5.0, 5.5)] + [on(bare, "label", 1.0)]
    spec = rule(parts={"label": "label"}, requirements=[needs("has", part="label")])
    test_scene = scene([labelled, bare], parts)

    [violation] = evaluate("r1", spec, test_scene)

    assert (violation.track_id, violation.t) == ("bare", 7.5)
    assert violation.clip == TimeSpan(start=4.5, end=8.5)
    assert checks(spec, test_scene) == (Check("bare", 7.5, 5.5), Check("labelled", 7.5, 5.5))


def test_lost_tracks_background_objects_and_items_never_seen_whole_are_not_judged() -> None:
    lost = track("lost", [(t / 2, 0.2 + t / 20) for t in range(8)])
    background = track("background", [(t / 2, 0.5 + (t % 2) / 100) for t in range(20)], y=0.2)
    # Runs along the top border, so it is never whole in the frame.
    cut_off = track("cut_off", [(t / 2, 0.2 + t / 20) for t in range(8)], y=0.1)
    spec = rule(parts={"label": "label"}, requirements=[needs("has", part="label")])
    test_scene = scene([lost, background, cut_off], [])

    assert evaluate("r1", spec, test_scene) == ()
    assert checks(spec, test_scene) == ()


def test_ever_before_accepts_the_part_at_any_point_before_the_checkpoint() -> None:
    a, b = crossing("a", y=0.3), crossing("b", y=0.7)
    spec = rule(parts={"cap": "cap"}, requirements=[needs("has", "ever_before", part="cap")])

    [violation] = evaluate("r1", spec, scene([a, b], [on(a, "cap", 2.0)]))

    assert violation.track_id == "b"
    assert violation.reason == "no cap before it left the view"


def test_count_at_least_counts_parts_seen_together_in_one_frame() -> None:
    four = crossing("four", y=0.3)
    three = crossing("three", y=0.7)
    offsets = (-0.06, -0.02, 0.02, 0.06)
    parts = [on(four, "flap", 3.0, dx) for dx in offsets]
    parts += [on(three, "flap", t, dx) for t in (3.0, 3.5) for dx in offsets[:3]]
    spec = rule(
        parts={"flap": "flap"},
        requirements=[needs("count_at_least", "ever_before", part="flap", n=4)],
    )

    [violation] = evaluate("r1", spec, scene([four, three], parts))

    assert violation.track_id == "three"
    assert violation.reason == "only 3 of 4 flap before it left the view"


def test_in_zone_checks_the_item_passed_through_a_drawn_area() -> None:
    washed, skipped = crossing("washed", y=0.3), crossing("skipped", y=0.7)
    wash = area("wash", 0.4, 0.2, 0.6, 0.4)
    spec = rule(requirements=[needs("in_zone", "ever_before", zone="wash")])

    [violation] = evaluate("r1", spec, scene([washed, skipped], [], (wash,)))

    assert violation.track_id == "skipped"
    assert violation.reason == "not in wash before it left the view"


@pytest.mark.parametrize(
    ("shape", "entered_at"),
    [
        (area("sealing", 0.58, 0.0, 0.8, 1.0).shape, 5.0),
        (Tripwire(start=Point(x=0.62, y=0.0), end=Point(x=0.62, y=1.0)), 5.2),
    ],
)
def test_enters_checks_when_the_item_enters_an_area_or_crosses_a_tripwire(
    shape: Polygon | Tripwire, entered_at: float
) -> None:
    labelled, unlabelled = crossing("labelled", y=0.3), crossing("unlabelled", y=0.7)
    sealing = Zone(key="sealing", shape=shape, source="drawn")
    parts = [on(labelled, "label", 4.5), on(unlabelled, "label", 2.0)]
    spec = rule(
        parts={"label": "label"},
        checkpoint={"kind": "enters", "zone": "sealing"},
        requirements=[needs("has", part="label")],
    )

    [violation] = evaluate("r1", spec, scene([labelled, unlabelled], parts, (sealing,)))

    assert violation.track_id == "unlabelled"
    assert violation.t == pytest.approx(entered_at)
    assert violation.reason == "no label when it entered sealing"


def test_end_of_footage_checks_every_item_as_last_seen() -> None:
    left, staying = crossing("left"), crossing("staying", start=22)
    spec = rule(
        parts={"label": "label"},
        checkpoint={"kind": "end_of_footage"},
        requirements=[needs("has", part="label")],
    )

    violations = evaluate("r1", spec, scene([left, staying], []))

    assert [(v.track_id, v.t) for v in violations] == [("left", 8.0), ("staying", 30.0)]
    assert violations[1].reason == "no label when the footage ended"


def test_order_flags_a_part_seen_before_the_part_that_must_come_first() -> None:
    in_order, reversed_, untaped = (
        crossing(name, y=y) for name, y in [("a", 0.2), ("b", 0.5), ("c", 0.8)]
    )
    parts = [on(in_order, "flap", 2.0), on(in_order, "tape", 4.0)]
    parts += [on(reversed_, "tape", 2.0), on(reversed_, "flap", 4.0), on(untaped, "flap", 2.0)]
    spec = rule(parts={"flap": "flap", "tape": "tape"}, order=[{"first": "flap", "then": "tape"}])

    [violation] = evaluate("r1", spec, scene([in_order, reversed_, untaped], parts))

    assert violation.track_id == "b"
    assert violation.reason == "tape came before flap"


def test_dwell_flags_an_item_that_stays_in_the_zone_too_long() -> None:
    moving = crossing("moving", y=0.3)
    stuck = track(
        "stuck", [(t, min(0.1 + 0.1 * t, 0.5)) for t in range(13)] + [(13, 0.7), (14, 0.9)]
    )
    capper = area("capper", 0.35, 0.0, 0.6, 1.0)
    spec = rule(dwell={"zone": "capper", "max_s": 5})

    [violation] = evaluate("r1", spec, scene([moving, stuck], [], (capper,)))

    assert (violation.track_id, violation.t) == ("stuck", 8.0)
    assert violation.reason == "stayed in capper for 9.0 s, more than 5 s"
    assert violation.clip == TimeSpan(start=3.0, end=9.0)


def test_rate_flags_each_whole_minute_below_the_minimum() -> None:
    slow_minute = [crossing(f"a{i}", start=5 * i) for i in range(10)]
    busy_minute = [crossing(f"b{i:02}", start=55 + 2 * i) for i in range(25)]
    spec = rule(rate={"min_per_minute": 20})

    [violation] = evaluate("r1", spec, scene(slow_minute + busy_minute, [], duration=150))

    assert (violation.track_id, violation.t, violation.clip) == (
        None,
        0.0,
        TimeSpan(start=0, end=60),
    )
    assert violation.reason == "10 jar passed between 0 s and 60 s, fewer than 20"


def test_violations_are_deterministic_with_stable_ids() -> None:
    items = [crossing(f"j{i}", start=i, y=0.2 + i / 10) for i in range(5)]
    parts = [on(items[0], "cap", 7.5), on(items[3], "cap", 4.0)]
    spec = rule(parts={"cap": "cap"}, requirements=[needs("has", part="cap")])

    first = evaluate("r1", spec, scene(items, parts))
    shuffled = evaluate("r1", spec, scene(items[::-1], parts[::-1]))

    assert first == shuffled
    assert len({v.id for v in first}) == len(first) == 4
    assert {v.id for v in evaluate("r2", spec, scene(items, parts))}.isdisjoint(v.id for v in first)


def test_predicates_dispatch_through_the_registry_by_kind() -> None:
    spec = rule(parts={"cap": "cap"}, requirements=[needs("has", part="cap")])
    always_fails = {**PREDICATES, "has": lambda predicate, item, span: "custom reason"}

    [violation] = evaluate("r1", spec, scene([crossing("a")], []), always_fails)

    assert violation.reason == "custom reason when it left the view"


def test_a_zone_that_is_missing_or_not_an_area_is_an_error() -> None:
    spec = rule(dwell={"zone": "capper", "max_s": 5})
    line = Zone(
        key="capper", shape=Tripwire(start=Point(x=0.5, y=0), end=Point(x=0.5, y=1)), source="drawn"
    )

    with pytest.raises(RuleError, match="not defined"):
        evaluate("r1", spec, scene([crossing("a")], []))
    with pytest.raises(RuleError, match="needs an area"):
        evaluate("r1", spec, scene([crossing("a")], [], (line,)))


def test_find_zone_asks_the_segmenter_for_the_phrase_on_one_keyframe() -> None:
    weak = at("capping machine", 3.0, 0.3, 0.3, 0.2).model_copy(update={"score": 0.4})
    strong = at("capping machine", 3.0, 0.5, 0.5, 0.2)
    segmenter = FakeSegmenter(lambda request: (weak, strong))
    keyframe = Keyframe(
        footage_id=FOOTAGE, image_key="keyframes/3.jpg", width=1280, height=720, t=3
    )

    found = asyncio.run(find_zone(segmenter, "capper", "capping machine", keyframe))
    missing = asyncio.run(find_zone(FakeSegmenter(lambda request: ()), "capper", "x", keyframe))

    expected = area("capper", 0.4, 0.4, 0.6, 0.6).shape
    assert found.zone == Zone(key="capper", shape=expected, source="text", phrase="capping machine")
    assert segmenter.requests[0].prompt == ConceptPrompt(text="capping machine")
    assert segmenter.requests[0].frame_times == (3.0,)
    assert missing.zone is None
    assert missing.charges
