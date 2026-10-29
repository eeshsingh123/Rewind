from collections.abc import Sequence

from rewind.contracts import Box, Observation, RuleSpec, TimeSpan, Track
from rewind.core.rules import Check, Scene, checks, evaluate

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


# A part seen on the item at time t (the item is sampled at that time).
def on(item: Track, concept: str, t: float) -> Observation:
    seen = next(o for o in item.observations if o.t == t)
    return at(concept, t, (seen.box.x0 + seen.box.x1) / 2, (seen.box.y0 + seen.box.y1) / 2, 0.04)


def label_rule() -> RuleSpec:
    return RuleSpec.model_validate(
        {
            "entity": {"key": "jar", "phrase": "jar"},
            "parts": {"label": "label"},
            "checkpoint": {"kind": "exits_view"},
            "requirements": [
                {"predicate": {"kind": "has", "part": "label"}, "timing": "at_checkpoint"}
            ],
        }
    )


def scene(tracks: list[Track], parts: list[Observation]) -> Scene:
    return Scene(FOOTAGE, 30, tuple(tracks), tuple(parts), ())


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
    test_scene = scene([labelled, bare], parts)

    [violation] = evaluate("r1", label_rule(), test_scene)

    assert (violation.track_id, violation.t) == ("bare", 7.5)
    assert violation.reason == "no label when it left the view"
    assert violation.clip == TimeSpan(start=4.5, end=8.5)
    assert checks(label_rule(), test_scene) == (
        Check("bare", 7.5, 5.5),
        Check("labelled", 7.5, 5.5),
    )


def test_lost_tracks_background_objects_and_items_never_seen_whole_are_not_judged() -> None:
    lost = track("lost", [(t / 2, 0.2 + t / 20) for t in range(8)])
    background = track("background", [(t / 2, 0.5 + (t % 2) / 100) for t in range(20)], y=0.2)
    # Runs along the top border, so it is never whole in the frame.
    cut_off = track("cut_off", [(t / 2, 0.2 + t / 20) for t in range(8)], y=0.1)
    test_scene = scene([lost, background, cut_off], [])

    assert evaluate("r1", label_rule(), test_scene) == ()
    assert checks(label_rule(), test_scene) == ()
