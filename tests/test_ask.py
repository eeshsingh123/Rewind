import json
from pathlib import Path

import pytest
from pydantic import JsonValue

from rewind.adapters.storage.local_disk import LocalDiskBlobStore
from rewind.adapters.storage.parquet_observations import ParquetObservationStore
from rewind.core.ask import (
    AskAgent,
    AskLimits,
    Finding,
    ProofClips,
    Sessions,
    Step,
    Toolset,
    Watcher,
)
from rewind.core.perception import Ingest, PerceptionRunner, ProxyMaker
from rewind.core.ports import ToolCall, ToolResultMessage, UserMessage
from tests.fakes import ScriptedModel, completion
from tests.test_analysis import FakeActivity, FakeSam, FakeVideo
from tests.test_insights import FakeSheet

pytestmark = pytest.mark.anyio

FIRST_LOOK = {
    "summary": "Two cartons ride a belt to the left.",
    "moments": [{"start_s": 1, "end_s": 2, "what": "A carton without a label passes."}],
    "questions": ["Which cartons have no label?"],
}
UNLABELLED = "The carton 1st from the left at 00:00 has no label."


def _call(name: str, arguments: dict[str, JsonValue]) -> ToolCall:
    return ToolCall(id=f"call-{name}", name=name, arguments=json.dumps(arguments))


# Upload, ask, watch it work: Muse watches once, then finds, looks, and answers with claims
# that code checks against what the tools found.
async def test_a_question_is_answered_from_tool_evidence_with_proof(tmp_path: Path) -> None:
    blobs = LocalDiskBlobStore(tmp_path, "http://localhost", b"test")
    video, sam, sheets = FakeVideo(), FakeSam(), FakeSheet()
    proxies = ProxyMaker(blobs, video, 720)
    watcher = Watcher(blobs, proxies)
    sessions = Sessions(blobs, Ingest(blobs, video, FakeActivity(), proxies, 2.0), watcher)
    runner = PerceptionRunner(proxies, ParquetObservationStore(blobs))
    tools = Toolset(blobs, watcher, runner, sheets, track_fps=2.0).tools()
    agent = AskAgent(sessions, tools, ProofClips(blobs, video, video), AskLimits())
    await blobs.write("uploads/belt.mp4", b"video")
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
    steps: list[Step] = []

    async def on_step(step: Step) -> None:
        steps.append(step)

    session, _ = await sessions.open("s1", "uploads/belt.mp4", model, on_step)
    answer = await agent.ask(session, "Which cartons have no label?", model, sam, on_step)

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
    assert finding == Finding(UNLABELLED, "seen", finding.clip_key)
    assert finding.clip_key is not None
    assert video.drawn[0][0].tag == "no label"
    saved = await sessions.get("s1")
    assert saved is not None
    assert [e.answer for e in saved.exchanges] == [UNLABELLED]

    # The same video again never pays for the same look twice.
    asked = len(model.requests)
    await sessions.open("s2", "uploads/belt.mp4", model, on_step)
    assert len(model.requests) == asked
