import json
from pathlib import Path

import pytest
from pydantic import JsonValue

from rewind.core.ask import Session, Step
from rewind.core.ask.evidence import Notebook
from rewind.core.ports import ToolCall, ToolResultMessage
from tests.fakes import FakeSam, ScriptedModel, completion
from tests.test_ask import FIRST_LOOK, Flow

pytestmark = pytest.mark.anyio

LABEL = "Does it have a label?"
TAPE = "Is it taped?"


def _call(name: str, arguments: dict[str, JsonValue], number: int = 1) -> ToolCall:
    return ToolCall(id=f"{name}-{number}", name=name, arguments=json.dumps(arguments))


def _sheet(*verdicts: str) -> str:
    tiles = [
        {"tile": str(n), "verdict": verdict, "answer": f"seen {verdict}"}
        for n, verdict in enumerate(verdicts, 1)
    ]
    return json.dumps({"tiles": tiles})


def _reading(**fields: object) -> dict[str, object]:
    empty = {"thing": "", "conditions": [], "combine": "none", "part": "", "plain": "",
             "unclear": "", "choices": []}  # fmt: skip
    return {**empty, **fields}


async def _session(flow: Flow, model: ScriptedModel) -> Session:
    await flow.blobs.write("uploads/belt.mp4", b"video")
    session, _ = await flow.sessions.open("s1", "uploads/belt.mp4", model, flow.on_step)
    return session


# Muse judges each item; code alone says which items meet both conditions, and the proof shows
# each of them.
async def test_code_combines_the_answers_into_an_exact_list(tmp_path: Path) -> None:
    flow = Flow(tmp_path, FakeSam())
    claim = "1 of 2 cartons has a label and no tape."
    combine = {
        "conditions": [{"question": LABEL, "want": "yes"}, {"question": TAPE, "want": "no"}],
        "how": "all",
        "describe": "labelled without tape",
    }
    model = ScriptedModel(
        [
            completion(json.dumps(FIRST_LOOK)),
            completion("", _call("look", {"question": LABEL, "thing": "carton"})),
            completion(_sheet("yes", "no")),
            completion("", _call("look", {"question": TAPE, "thing": "carton"}, 2)),
            completion(_sheet("no", "no")),
            completion("", _call("combine", combine)),
            completion("", _call("answer", {"claims": [{"text": claim, "evidence": ["e7"]}]})),
        ],
        readings=[_reading(plain="Checking cartons: label and no tape, together.")],
    )
    session = await _session(flow, model)

    answer = await flow.agent.ask(session, "Which?", model, flow.segmenters, flow.on_step)

    combined = model.requests[-1].messages[-1]
    assert isinstance(combined, ToolResultMessage)
    assert "Labelled without tape: 1 of 2" in combined.text
    [finding] = answer.findings
    assert finding.proof_key is not None
    assert finding.proof_key.endswith(".jpg")
    assert len(flow.sheets.tiles[-1]) == 1
    assert answer.reading == "Checking cartons: label and no tape, together."


async def test_an_item_that_cannot_be_judged_gets_a_second_view(tmp_path: Path) -> None:
    flow = Flow(tmp_path, FakeSam())
    claim = {"text": "One carton has no label.", "evidence": ["e3"]}
    model = ScriptedModel(
        [
            completion(json.dumps(FIRST_LOOK)),
            completion("", _call("look", {"question": LABEL, "thing": "carton"})),
            completion(_sheet("cannot tell", "yes")),
            completion(_sheet("no")),
            completion("", _call("answer", {"claims": [claim]})),
        ]
    )
    session = await _session(flow, model)

    await flow.agent.ask(session, "Labels?", model, flow.segmenters, flow.on_step)

    first_view, second_view = flow.sheets.tiles[-2][0], flow.sheets.tiles[-1][0]
    assert first_view.t != second_view.t
    assert len(flow.sheets.tiles[-1]) == 1
    looked = model.requests[-1].messages[-1]
    assert isinstance(looked, ToolResultMessage)
    assert "seen no (no)" in looked.text


# New words for something already tracked must not pay SAM again.
async def test_another_wording_reuses_what_was_found(tmp_path: Path) -> None:
    sam = FakeSam()
    flow = Flow(tmp_path, sam)
    claim = {"text": "Two cartons came past in the clip.", "evidence": ["e2"]}
    model = ScriptedModel(
        [
            completion(json.dumps(FIRST_LOOK)),
            completion("", _call("find", {"thing": "carton"})),
            completion("", _call("answer", {"claims": [claim]})),
            completion("", _call("find", {"thing": "cartons on the belt"})),
            completion("", _call("answer", {"claims": [claim]})),
        ]
    )
    session = await _session(flow, model)

    await flow.agent.ask(session, "How many?", model, flow.segmenters, flow.on_step)
    saved = await flow.sessions.get("s1")
    assert saved is not None
    assert saved.things == ("carton",)
    tracked = sam.calls
    await flow.agent.ask(saved, "On the belt?", model, flow.segmenters, flow.on_step)

    assert sam.calls == tracked


async def test_a_question_with_two_meanings_is_asked_back_once(tmp_path: Path) -> None:
    flow = Flow(tmp_path, FakeSam())
    unclear = _reading(
        unclear="The right of the belt, or flaps opening right?",
        choices=["The right of the belt", "Flaps opening right"],
    )
    say = "The carton on the right is open."
    model = ScriptedModel(
        [
            completion(json.dumps(FIRST_LOOK)),
            completion("", _call("answer", {"claims": [{"text": say, "evidence": ["e2"]}]})),
        ],
        readings=[unclear, unclear],
    )
    session = await _session(flow, model)

    back = await flow.agent.ask(session, "Open right?", model, flow.segmenters, flow.on_step)

    assert back.answer == "The right of the belt, or flaps opening right?"
    assert back.choices == ("The right of the belt", "Flaps opening right")
    assert back.findings == ()
    saved = await flow.sessions.get("s1")
    assert saved is not None
    answered = await flow.agent.ask(
        saved, "The right of the belt", model, flow.segmenters, flow.on_step
    )
    assert [finding.text for finding in answered.findings] == [say]


# The list code worked out is an answer in itself, whatever the agent then does with it.
async def test_a_combined_list_survives_an_agent_that_never_answers(tmp_path: Path) -> None:
    flow = Flow(tmp_path, FakeSam())
    combine = {"conditions": [{"question": LABEL, "want": "no"}], "describe": "no label"}
    model = ScriptedModel(
        [
            completion(json.dumps(FIRST_LOOK)),
            completion("", _call("look", {"question": LABEL, "thing": "carton"})),
            completion(_sheet("yes", "no")),
            completion("", _call("combine", combine)),
            completion("Two boxes do."),
            completion("Two boxes do."),
        ]
    )
    session = await _session(flow, model)

    answer = await flow.agent.ask(session, "Which?", model, flow.segmenters, flow.on_step)

    [finding] = answer.findings
    assert finding.text.startswith("No label: 1 of 2")
    assert finding.proof_key is not None


async def test_a_claim_that_cannot_be_backed_is_mentioned(tmp_path: Path) -> None:
    flow = Flow(tmp_path, FakeSam())
    good = {"text": "One carton passes with no label.", "evidence": ["e2"]}
    bad = {"text": "9 cartons went past today.", "evidence": ["e2"]}
    model = ScriptedModel(
        [
            completion(json.dumps(FIRST_LOOK)),
            completion("", _call("answer", {"claims": [good, bad]})),
            completion("", _call("answer", {"claims": [good, bad]})),
        ]
    )
    session = await _session(flow, model)

    answer = await flow.agent.ask(session, "Anything?", model, flow.segmenters, flow.on_step)

    assert [finding.text for finding in answer.findings] == [good["text"]]
    assert answer.left_out == "Left out 1 more point I could not back with proof."


async def test_the_wait_for_tracking_is_shown(tmp_path: Path) -> None:
    flow = Flow(tmp_path, FakeSam())
    claim = {"text": "Two cartons came past in the clip.", "evidence": ["e2"]}
    model = ScriptedModel(
        [
            completion(json.dumps(FIRST_LOOK)),
            completion("", _call("find", {"thing": "carton"})),
            completion("", _call("answer", {"claims": [claim]})),
        ]
    )
    session = await _session(flow, model)

    await flow.agent.ask(session, "How many?", model, flow.segmenters, flow.on_step)

    assert Step(text="Tracking every carton, under a minute", done=False) in flow.steps


def test_a_count_of_the_items_cited_is_traceable() -> None:
    notebook = Notebook()
    first = notebook.add("measured", "The box at 00:08 stood still.")
    second = notebook.add("measured", "The box at 00:16 stood still.")

    assert not isinstance(notebook.check("2 boxes stood still.", [first.id, second.id]), str)
