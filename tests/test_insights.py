import json
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rewind.adapters.storage.local_disk import LocalDiskBlobStore
from rewind.adapters.storage.sql import SqlRepository
from rewind.contracts import (
    Entity,
    ExitsView,
    Footage,
    Has,
    Line,
    Requirement,
    Rule,
    RuleSpec,
    TimeSpan,
    Violation,
)
from rewind.core.analysis import Insights, InsightWriter, Tile
from rewind.core.analysis.facts import Fact, untraceable
from rewind.core.analysis.patterns import patterns
from rewind.core.analysis.timeline import Judged, Timeline, find_stops
from rewind.core.ports import UserMessage
from tests.fakes import ScriptedModel, completion

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 10, 29, 9, 0, tzinfo=UTC)
JAR_RULE = RuleSpec(
    entity=Entity(key="jar", phrase="jar"),
    parts={"label": "label"},
    checkpoint=ExitsView(),
    requirements=(Requirement(predicate=Has(part="label"), timing="at_checkpoint"),),
)


def _items(count: int, failed: Sequence[int], every_s: float = 2.0) -> tuple[Judged, ...]:
    return tuple(
        Judged(rule_id="rule-1", track_id=f"jar#{i}", t=i * every_s,
               violation_id=f"v{i}" if i in failed else None)
        for i in range(count)
    )  # fmt: skip


def test_failures_in_a_row_stand_out_and_spread_ones_do_not() -> None:
    clustered = patterns(_items(30, [20, 21, 22, 23]), ())
    spread = patterns(_items(30, [2, 9, 19, 26]), ())

    assert [p.kind for p in clustered if p.notable] == ["in_a_row"]
    assert [p.numbers for p in clustered if p.kind == "in_a_row"] == [(4,)]
    assert not [p for p in spread if p.notable]


def test_a_failure_on_every_nth_item_points_at_one_head() -> None:
    found = {p.kind: p for p in patterns(_items(30, [2, 8, 14, 20, 26]), ()) if p.notable}

    assert found["every_nth"].numbers == (6, 3, 9, 15, 21, 27)


def _gap_in_output() -> list[Judged]:
    times = [*range(0, 21, 2), *range(35, 61, 2)]
    return [Judged(rule_id="r", track_id=f"jar#{t}", t=float(t)) for t in times]


def test_a_long_gap_in_output_where_nothing_moved_is_a_stop_and_says_what_it_cost() -> None:
    moving = (TimeSpan(start=0, end=21.5), TimeSpan(start=33.0, end=60.0))

    [stop] = find_stops(_gap_in_output(), moving, 60.0)

    assert (stop.gap.start, stop.gap.end) == (20.0, 35.0)
    assert (stop.stood, stop.seen) == (TimeSpan(start=21.5, end=33.0), "no_motion")
    assert stop.usual_gap_s == 2.0
    assert stop.lost_items == 6


def test_items_standing_still_make_a_stop_and_a_gap_without_evidence_does_not() -> None:
    def speed(t: float) -> float:
        return 0.0 if 24 <= t < 32 else 0.05

    # Two items in view the whole time, moving with the belt except while it stood.
    paths = [[(t / 2, x, 0.5) for t, x in zip(range(121), _positions(speed, start), strict=True)]
             for start in (0.1, 0.4)]  # fmt: skip
    always_moving = (TimeSpan(start=0, end=60.0),)

    [held] = find_stops(_gap_in_output(), always_moving, 60.0, paths)
    unproven = find_stops(_gap_in_output(), always_moving, 60.0)

    assert (held.stood, held.seen) == (TimeSpan(start=24.0, end=32.0), "items_still")
    assert unproven == ()


def _positions(speed: Callable[[float], float], start: float) -> list[float]:
    x, found = start, []
    for step in range(121):
        found.append(x)
        x += speed(step / 2) * 0.5
    return found


def test_a_claim_with_a_number_its_facts_do_not_state_is_caught() -> None:
    facts = [Fact("F2", "Stop: no jar came through from 00:20 to 00:35 (15 s); about 6 jars lost.")]

    assert untraceable("The line stopped at 0:20 for 15 s, about six jars lost.", facts) == ()
    assert untraceable("The line stopped for 15 s and lost 7 jars at 00:21.", facts) == (
        "7", "00:21"
    )  # fmt: skip


class FakeSheet:
    def __init__(self) -> None:
        self.tiles: list[tuple[Tile, ...]] = []

    async def make_sheet(self, source: Path, destination: Path, tiles: Sequence[Tile]) -> None:
        self.tiles.append(tuple(tiles))
        destination.write_bytes(b"\xff\xd8 jpeg")


class World:
    def __init__(self, repository: SqlRepository, blobs: LocalDiskBlobStore) -> None:
        self.repository = repository
        self.blobs = blobs
        self.model = ScriptedModel([])
        self.sheet = FakeSheet()
        self.insights = Insights(
            repository, blobs, self.sheet, lambda _: InsightWriter(self.model, blobs)
        )
        self.line = Line(id="line-1", workspace_id="ws", name="Jar line", created_at=NOW)

    async def footage(self, footage_id: str, judged: Sequence[Judged], stops: bool) -> Footage:
        footage = Footage(
            id=footage_id, line_id=self.line.id, content_hash=footage_id,
            original_key=f"footage/{footage_id}/belt.mp4", duration_s=60.0, fps=25.0,
            width=1280, height=720, status="ready",
        )  # fmt: skip
        await self.blobs.write(footage.original_key, b"video")
        await self.repository.save_footage(footage)
        moving = (TimeSpan(start=0, end=21.5), TimeSpan(start=33.0, end=60.0))
        found = find_stops(judged, moving, 60.0) if stops else ()
        await self.insights.save_timeline(
            Timeline(footage_id=footage_id, duration_s=60.0, judged=tuple(judged), stops=found)
        )
        violations = [
            Violation(id=j.violation_id, rule_id=j.rule_id, footage_id=footage_id,
                      track_id=j.track_id, t=j.t, reason="no label when it left the view",
                      clip=TimeSpan(start=j.t - 3, end=j.t + 1))
            for j in judged if j.violation_id
        ]  # fmt: skip
        await self.repository.replace_violations(footage_id, ["rule-1"], violations)
        return footage


@pytest.fixture
async def world(tmp_path: Path) -> AsyncIterator[World]:
    repository = SqlRepository.from_url(f"sqlite+aiosqlite:///{tmp_path / 'rewind.db'}")
    await repository.create_tables()
    world = World(repository, LocalDiskBlobStore(tmp_path / "blobs", "https://t.test", b"k"))
    await repository.save_line(world.line)
    await repository.save_rule(
        Rule(id="rule-1", line_id=world.line.id, text="Every jar has a label when it leaves.",
             normalized_text="every jar has a label when it leaves", spec=JAR_RULE,
             created_at=NOW)
    )  # fmt: skip
    yield world
    await repository.close()


def _with_stop(failed: Sequence[int]) -> list[Judged]:
    times = [*range(0, 21, 2), *range(35, 61, 2)]
    return [Judged(rule_id="rule-1", track_id=f"jar#{t}", t=float(t),
                   violation_id=f"v{t}" if t in failed else None) for t in times]  # fmt: skip


async def test_a_normal_run_gets_one_quiet_line_and_no_model_call(world: World) -> None:
    footage = await world.footage("f1", _items(25, [3, 12, 21]), stops=False)

    found = await world.insights.find("ws", world.line, footage, {"rule-1": 1})

    assert found.insights == ()
    assert found.quiet == (
        "Nothing else stood out: the line didn't stop, and the jars that failed look spread out."
    )
    assert world.model.requests == []


async def test_muse_words_what_stands_out_and_claims_it_cannot_trace_are_dropped(
    world: World,
) -> None:
    await world.footage("f0", _items(20, []), stops=False)
    footage = await world.footage("f1", _with_stop([36, 38]), stops=True)
    world.model.replies = [
        completion(json.dumps({"insights": [
            {"text": "The line stood still from 00:21 to 00:33, and about 6 jars of output were "
                     "lost.", "facts": ["F2"], "clip": "F2", "label": "The stop"},
            {"text": "Labels ran out at 00:30, costing 12 jars.", "facts": ["F2"], "clip": "F2",
             "label": "Labels"},
            {"text": "Made up.", "facts": ["F99"], "clip": "F99", "label": "?"},
        ]}))
    ]  # fmt: skip

    found = await world.insights.find("ws", world.line, footage, {"rule-1": 1})
    again = await world.insights.find("ws", world.line, footage, {"rule-1": 1})

    [insight] = found.insights
    assert insight.label == "The stop"
    assert insight.clip == TimeSpan(start=19.5, end=35.0)
    assert insight.facts[0].startswith("Stop: from 00:21 to 00:33 (12 s) nothing moved")
    # The same facts never cost a second call.
    assert again.insights == found.insights
    [request] = world.model.requests
    sheet = request.messages[1]
    assert isinstance(sheet, UserMessage)
    assert "rule 1 on 1 other recording of this line: 0 of 20 jars failed" in sheet.text.lower()
    assert sheet.image_keys
    assert world.sheet.tiles[0][0].label == "F2 00:27"
    saved = await world.insights.saved(footage.id)
    assert saved.get(insight.id) == insight


async def test_a_question_the_footage_cannot_answer_says_so_and_suggests_a_rule(
    world: World,
) -> None:
    footage = await world.footage("f1", _with_stop([36, 38]), stops=True)
    world.model.replies = [
        completion(json.dumps({
            "answerable": False, "answer": "I can't tell who stopped the line from the footage.",
            "facts": [], "clip": "", "label": "", "suggest_rule": "",
        })),
        completion(json.dumps({
            "answerable": True, "answer": "It stopped for 14 s.", "facts": ["F2"], "clip": "F2",
            "label": "The stop",
        "suggest_rule": ""})),
        completion(json.dumps({
            "answerable": True, "answer": "The line was stopped for 12 s, from 00:21.",
            "facts": ["F2"], "clip": "F2", "label": "The stop", "suggest_rule": "",
        })),
    ]  # fmt: skip

    who = await world.insights.ask("ws", world.line, footage, {"rule-1": 1}, "who did it?", None)
    how_long = await world.insights.ask(
        "ws", world.line, footage, {"rule-1": 1}, "how long did it stop?", None
    )

    assert who is not None
    assert who.insight is None
    assert who.text == "I can't tell who stopped the line from the footage."
    # "14 s" is in no fact, so the answer was sent back once and corrected.
    assert how_long is not None
    assert how_long.text == "The line was stopped for 12 s, from 00:21."
    assert how_long.insight is not None
    assert how_long.insight.label == "The stop"
    assert "not in the facts you cited: 14" in world.model.requests[2].messages[-1].text
