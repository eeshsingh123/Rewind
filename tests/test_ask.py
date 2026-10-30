import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import JsonValue

from rewind.adapters.storage.local_disk import LocalDiskBlobStore
from rewind.adapters.storage.parquet_observations import ParquetObservationStore
from rewind.contracts import Observation
from rewind.core.analysis import Mark, PartLooker
from rewind.core.ask import (
    AskAgent,
    AskLimits,
    Proofs,
    Segmenters,
    Sessions,
    Step,
    Toolset,
    Watcher,
)
from rewind.core.perception import Ingest, PerceptionRunner, ProxyMaker
from rewind.core.ports import Segmentation, SegmentRequest, ToolCall, ToolResultMessage, UserMessage
from tests.fakes import ScriptedModel, completion
from tests.test_analysis import FakeActivity, FakeSam, FakeVideo, _frames, _seen
from tests.test_insights import FakeSheet

pytestmark = pytest.mark.anyio

FIRST_LOOK = {
    "summary": "Two cartons ride a belt to the left.",
    "moments": [{"start_s": 1, "end_s": 2, "what": "A carton without a label passes."}],
    "questions": ["Which cartons have no label?"],
}
UNLABELLED = "The carton 1st from the left at 00:00 has no label."
PHOTO_LOOK = {
    "summary": "Three boxes stand on a table.",
    "notes": ["The box on the right looks open."],
    "questions": ["Does each box have a label on top?"],
}


def _call(name: str, arguments: dict[str, JsonValue]) -> ToolCall:
    return ToolCall(id=f"call-{name}", name=name, arguments=json.dumps(arguments))


class FakeMedia(FakeVideo):
    def __init__(self) -> None:
        super().__init__()
        self.photos: list[tuple[Mark, ...]] = []

    async def upright(self, source: Path, destination: Path) -> tuple[int, int]:
        destination.write_bytes(b"\xff\xd8 upright")
        return 1200, 800

    async def make_photo(self, source: Path, destination: Path, marks: Sequence[Mark]) -> None:
        self.photos.append(tuple(marks))
        destination.write_bytes(b"\xff\xd8 boxes")


# Three boxes side by side on one photo; SAM lists them in no particular order.
class FakeImageSam(FakeSam):
    async def segment(self, request: SegmentRequest) -> Segmentation:
        self.calls += 1
        self.prompts.append(getattr(request.prompt, "text", ""))
        found: list[Observation] = []
        if self.prompts[-1] == "box":
            found = [_seen(request, "box", 0.0, i, x, 0.2) for i, x in enumerate((0.6, 0.1, 0.35))]
        return Segmentation(tuple(found), (_frames(1),))


@dataclass
class Flow:
    tmp_path: Path
    sam: FakeSam
    media: FakeMedia = field(default_factory=FakeMedia)
    sheets: FakeSheet = field(default_factory=FakeSheet)
    steps: list[Step] = field(default_factory=list[Step])

    def __post_init__(self) -> None:
        blobs = LocalDiskBlobStore(self.tmp_path, "http://localhost", b"test")
        self.blobs = blobs
        proxies = ProxyMaker(blobs, self.media, 720)
        watcher = Watcher(blobs, proxies)
        ingest = Ingest(blobs, self.media, FakeActivity(), proxies, 2.0)
        self.sessions = Sessions(blobs, ingest, watcher, self.media, self.media)
        store = ParquetObservationStore(blobs)
        stills = PartLooker(blobs, store, self.media)
        runner = PerceptionRunner(proxies, store)
        tools = Toolset(blobs, watcher, runner, stills, self.sheets, track_fps=2.0).tools()
        proofs = Proofs(blobs, self.media, self.media)
        self.agent = AskAgent(self.sessions, tools, proofs, AskLimits())
        self.segmenters = Segmenters(video=self.sam, image=self.sam)

    async def on_step(self, step: Step) -> None:
        self.steps.append(step)


# Upload, ask, watch it work: Muse watches once, then finds, looks, and answers with claims
# that code checks against what the tools found.
async def test_a_question_is_answered_from_tool_evidence_with_proof(tmp_path: Path) -> None:
    sam = FakeSam()
    flow = Flow(tmp_path, sam)
    sessions, sheets, video, steps = flow.sessions, flow.sheets, flow.media, flow.steps
    await flow.blobs.write("uploads/belt.mp4", b"video")
    model = ScriptedModel([
        completion(json.dumps(FIRST_LOOK)),
        completion("", _call("find", {"thing": "Carton"})),
        completion("", _call("look", {"question": "Does it have a label?", "thing": "carton"})),
        completion(json.dumps({"tiles": [{"tile": "1", "answer": "no label"},
                                         {"tile": "2", "answer": "has a label"}]})),
        completion("", _call("answer", {"claims": [
            {"text": UNLABELLED, "evidence": ["e4"]},
            {"text": "3 cartons went out today.", "evidence": ["e3"]},
        ]})),
        completion("", _call("answer", {"claims": [{"text": UNLABELLED, "evidence": ["e4"]}]})),
    ])  # fmt: skip

    session, _ = await sessions.open("s1", "uploads/belt.mp4", model, flow.on_step)
    question = "Which cartons have no label?"
    answer = await flow.agent.ask(session, question, model, flow.segmenters, flow.on_step)

    first = model.requests[0].messages[-1]
    assert isinstance(first, UserMessage)
    assert first.video_key is not None
    assert session.first_look.questions == ("Which cartons have no label?",)
    assert [s.text for s in steps if s.done] == [
        "Watched the whole video", "Found 2 cartons", "Looked closer at 2 cartons"
    ]  # fmt: skip
    # The look reused the cartons find tracked: SAM was paid once.
    assert sam.calls == 1
    assert [tile.box is not None for tile in sheets.tiles[0]] == [True, True]
    # "3" is in no evidence, so that claim went back to Muse, and was left out.
    sent_back = model.requests[-1].messages[-1]
    assert isinstance(sent_back, ToolResultMessage)
    assert "not in the evidence cited: 3" in sent_back.text
    [finding] = answer.findings
    assert (finding.text, finding.kind) == (UNLABELLED, "seen")
    assert finding.proof_key is not None
    assert finding.span is not None
    assert video.drawn[0][0].tag == "no label"
    saved = await sessions.get("s1")
    assert saved is not None
    assert [e.answer for e in saved.exchanges] == [UNLABELLED]

    # The same video again never pays for the same look twice.
    asked = len(model.requests)
    await sessions.open("s2", "uploads/belt.mp4", model, flow.on_step)
    assert len(model.requests) == asked


# A photo goes through the same doors: Muse looks once, SAM finds items by image, Muse judges
# each one, and the proof is the photo with the box drawn.
async def test_a_photo_is_asked_about_like_a_video(tmp_path: Path) -> None:
    sam = FakeImageSam()
    flow = Flow(tmp_path, sam)
    await flow.blobs.write("uploads/table.jpg", b"photo")
    no_label = "The 2nd box from the left has no label on top."
    model = ScriptedModel([
        completion(json.dumps(PHOTO_LOOK)),
        completion("", _call("find", {"thing": "box"})),
        completion("", _call("look", {"question": "Is there a label on top?", "thing": "box"})),
        completion(json.dumps({"tiles": [{"tile": "1", "answer": "has a label"},
                                         {"tile": "2", "answer": "no label"},
                                         {"tile": "3", "answer": "has a label"}]})),
        completion("", _call("measure", {"thing": "box"})),
        completion("", _call("answer", {"claims": [{"text": no_label, "evidence": ["e5"]}]})),
    ])  # fmt: skip

    session, _ = await flow.sessions.open("p1", "uploads/table.jpg", model, flow.on_step)
    question = "Does each box have a label on top?"
    answer = await flow.agent.ask(session, question, model, flow.segmenters, flow.on_step)

    assert session.photo
    first = model.requests[0].messages[-1]
    assert isinstance(first, UserMessage)
    assert first.video_key is None
    assert first.image_keys == (session.footage.original_key,)
    assert session.first_look.moments[0].what == "The box on the right looks open."
    agent_turn = model.requests[1]
    assert [tool.name for tool in agent_turn.tools] == [
        "find", "look", "measure", "area", "answer"
    ]  # fmt: skip
    assert [s.text for s in flow.steps if s.done] == [
        "Looked at the photo", "Found 3 boxes", "Looked closer at 3 boxes",
        "Counted 3 boxes in the photo",
    ]  # fmt: skip
    # find, look and measure all reused one image call.
    assert sam.calls == 1
    found = model.requests[2].messages[-1]
    assert isinstance(found, ToolResultMessage)
    assert "the 2nd box from the left and the 3rd box from the left" in found.text
    assert [tile.t for tile in flow.sheets.tiles[0]] == [0.0, 0.0, 0.0]
    [finding] = answer.findings
    assert finding.kind == "seen"
    assert finding.proof_key is not None
    assert finding.proof_key.endswith(".jpg")
    [[drawn]] = flow.media.photos
    assert (drawn.kind, drawn.tag) == ("flagged", "no label")
    assert drawn.path[0].box.x0 == 0.35

    asked = len(model.requests)
    await flow.sessions.open("p2", "uploads/table.jpg", model, flow.on_step)
    assert len(model.requests) == asked


# Muse marks a part of a frame, SAM confirms it, and code counts only what is inside. A part
# that cannot be confirmed is reported as such, with no box.
async def test_a_question_about_one_area_counts_only_inside_it(tmp_path: Path) -> None:
    sam = FakeSam()
    flow = Flow(tmp_path, sam)
    await flow.blobs.write("uploads/belt.mp4", b"video")
    top_half = {"object_name": "the top half", "holds": "carton",
                "bbox": {"x_min": 0, "y_min": -2, "x_max": 1001, "y_max": 450}}  # fmt: skip
    wall = {"object_name": "the back wall", "holds": "wall",
            "bbox": {"x_min": 0, "y_min": 0, "x_max": 1000, "y_max": 200}}  # fmt: skip
    counted = "1 carton in the top half."
    unseen = "The back wall could not be made out."
    model = ScriptedModel([
        completion(json.dumps(FIRST_LOOK)),
        completion("", _call("area", {"name": "The top half", "at_s": 1.0})),
        completion(json.dumps({"areas": [top_half]})),
        completion("", _call("measure", {"thing": "carton", "area": "the top half"})),
        completion("", _call("answer", {"claims": [{"text": counted, "evidence": ["e4"]}]})),
        completion("", _call("area", {"name": "the back wall", "at_s": 1.0})),
        completion(json.dumps({"areas": [wall]})),
        completion("", _call("answer", {"claims": [{"text": unseen, "evidence": ["e3"]}]})),
    ])  # fmt: skip

    session, _ = await flow.sessions.open("s1", "uploads/belt.mp4", model, flow.on_step)
    question = "How many cartons are in the top half?"
    answer = await flow.agent.ask(session, question, model, flow.segmenters, flow.on_step)

    marking = model.requests[2].messages[-1]
    assert isinstance(marking, UserMessage)
    assert len(marking.image_keys) == 1
    [finding] = answer.findings
    assert (finding.text, finding.kind) == (counted, "measured")
    assert finding.proof_key is not None
    area, carton = flow.media.drawn[-1]
    assert (area.kind, area.tag) == ("area", "the top half")
    assert (area.path[0].box.y0, area.path[0].box.x1) == (0.0, 1.0)
    assert carton.kind == "passed"
    assert all(s.box.y0 < 0.45 for s in carton.path)

    session = await flow.sessions.get("s1") or session
    drawn = len(flow.media.drawn)
    answer = await flow.agent.ask(
        session, "Is the back wall clean?", model, flow.segmenters, flow.on_step
    )
    [finding] = answer.findings
    assert (finding.text, finding.kind, finding.proof_key) == (unseen, "seen", None)
    assert len(flow.media.drawn) == drawn
    assert "Could not make out the back wall" in [s.text for s in flow.steps if s.done][-1]
