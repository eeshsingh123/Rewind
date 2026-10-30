import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from PIL import Image
from pillow_heif import register_heif_opener
from pydantic import JsonValue

from rewind.adapters.video.photo import PillowPhotos
from rewind.app.web import mount_page, web_router
from rewind.core.ask import Runs, Sample, Samples
from rewind.core.budget import BudgetExceededError
from rewind.core.ports import (
    Completion,
    CompletionRequest,
    ObjectPrompts,
    Segmentation,
    SegmentRequest,
)
from tests.fakes import FakeSam, ScriptedModel, billed_frames, completion, seen
from tests.test_ask import FIRST_LOOK, PHOTO_LOOK, FakeImageSam, Flow, _call

pytestmark = pytest.mark.anyio

QUESTION = "Which cartons have no label?"
NO_LABEL = "The carton 1st from the left at 00:00 has no label."
BOX = {"x0": 0.1, "y0": 0.2, "x1": 0.3, "y1": 0.5}


# Follows one object across the frame from a box; anything else is the usual two cartons.
class FollowingSam(FakeSam):
    async def segment(self, request: SegmentRequest) -> Segmentation:
        if not isinstance(request.prompt, ObjectPrompts):
            return await super().segment(request)
        self.calls += 1
        self.prompts.append(request.prompt.concept)
        found = [
            seen(request, request.prompt.concept, t, 0, 0.1 + 0.05 * t, 0.3)
            for t in request.frame_times
            if 2 <= t <= 6
        ]
        return Segmentation(tuple(found), (billed_frames(len(request.frame_times)),))


class SpentModel(ScriptedModel):
    async def complete(self, request: CompletionRequest) -> None:  # type: ignore[override]
        raise BudgetExceededError("daily cap")


@asynccontextmanager
async def serving(
    flow: Flow, model: ScriptedModel, *samples: Sample
) -> AsyncIterator[httpx.AsyncClient]:
    runs = Runs(flow.sessions, flow.agent, lambda *_: (model, flow.segmenters))
    shelf = Samples(samples, flow.blobs, flow.media, flow.media)
    app = FastAPI()
    app.include_router(web_router(runs, flow.sessions, shelf, flow.blobs))
    mount_page(app)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    await runs.close()


async def _settled(client: httpx.AsyncClient, session_id: str) -> dict[str, object]:
    for _ in range(200):
        live = (await client.get(f"/api/sessions/{session_id}/live")).json()
        if not live["working"]:
            return live
        await asyncio.sleep(0.01)
    raise AssertionError("the run never finished")


async def _upload(client: httpx.AsyncClient, name: str, data: bytes = b"footage") -> str:
    response = await client.post("/api/sessions", files={"file": (name, data)})
    assert response.status_code == 201
    return response.json()["id"]


def _answering(claim: dict[str, JsonValue]) -> list[Completion]:
    return [
        completion("", _call("find", {"thing": "carton"})),
        completion("", _call("look", {"question": "Does it have a label?", "thing": "carton"})),
        completion(json.dumps({"tiles": [{"tile": "1", "verdict": "no", "answer": "no label"},
                                         {"tile": "2", "verdict": "yes",
                                          "answer": "has a label"}]})),
        completion("", _call("answer", {"claims": [claim]})),
    ]  # fmt: skip


# Upload in the browser, ask, see each step, and open the same link later: the answer, its proof
# and its cost are all still there, and the proof plays through a signed link.
async def test_upload_ask_and_come_back_to_the_link(tmp_path: Path) -> None:
    flow = Flow(tmp_path, FakeSam())
    model = ScriptedModel([
        completion(json.dumps(FIRST_LOOK)),
        *_answering({"text": NO_LABEL, "evidence": ["e4"]}),
    ])  # fmt: skip
    async with serving(flow, model) as client:
        session_id = await _upload(client, "IMG_0001.MOV")
        await _settled(client, session_id)
        opened = (await client.get(f"/api/sessions/{session_id}")).json()
        assert opened["summary"] == FIRST_LOOK["summary"]
        assert [m["start_s"] for m in opened["moments"]] == [1]

        asked = await client.post(
            f"/api/sessions/{session_id}/questions", json={"question": QUESTION}
        )
        assert asked.status_code == 202
        busy = await client.post(
            f"/api/sessions/{session_id}/questions", json={"question": "Again?"}
        )
        assert busy.status_code == 409
        live = await _settled(client, session_id)
        assert [s["text"] for s in live["steps"] if s["done"]] == [  # type: ignore[union-attr]
            "Found 2 cartons", "Looked closer at 2 cartons"
        ]  # fmt: skip

        again = (await client.get(f"/api/sessions/{session_id}")).json()
        [exchange] = again["exchanges"]
        [finding] = exchange["findings"]
        assert finding["text"] == NO_LABEL
        assert (finding["kind"], finding["proof_photo"]) == ("seen", False)
        assert finding["start_s"] is not None
        assert float(exchange["cost_usd"]) > 0
        assert exchange["steps"] == ["Found 2 cartons", "Looked closer at 2 cartons"]

        played = await client.get(finding["proof_url"], headers={"Range": "bytes=0-3"})
        assert played.status_code == 206
        assert played.headers["content-type"] == "video/mp4"
        assert (await client.get(again["media_url"])).content == b"playable"
        query = parse_qs(urlsplit(finding["proof_url"]).query)
        forged = f"{urlsplit(finding['proof_url']).path}?expires={query['expires'][0]}&sig=forged"
        assert (await client.get(forged)).status_code == 403
        assert (
            (await client.get(f"/s/{session_id}")).headers["content-type"].startswith("text/html")
        )
        assert (await client.get("/static/app.js")).status_code == 200


# A sample's first look and first answer are already known, so tapping it costs nothing, and the
# visitor can still ask their own question, which runs the real agent.
async def test_a_sample_answers_at_once_for_free(tmp_path: Path) -> None:
    clip = tmp_path / "belt.mp4"
    clip.write_bytes(b"sample video")
    flow = Flow(tmp_path / "blobs", FakeSam())
    model = ScriptedModel([
        completion(json.dumps(FIRST_LOOK)),
        *_answering({"text": NO_LABEL, "evidence": ["e4"]}),
        completion("", _call("answer", {"claims": [{"text": "Nothing else stands out.",
                                                    "evidence": ["e2"]}]})),
    ])  # fmt: skip
    sample = Sample("belt", "A belt", clip)
    async with serving(flow, model, sample) as client:
        [listed] = (await client.get("/api/samples")).json()
        assert (listed["name"], listed["photo"]) == ("belt", False)
        assert (await client.get(listed["picture_url"])).status_code == 200

        primed = (await client.post("/api/samples/belt")).json()["id"]
        await _settled(client, primed)
        await client.post(f"/api/sessions/{primed}/questions", json={"question": QUESTION})
        await _settled(client, primed)
        paid = len(model.requests)

        tapped = (await client.post("/api/samples/belt")).json()["id"]
        await _settled(client, tapped)
        view = (await client.get(f"/api/sessions/{tapped}")).json()
        [exchange] = view["exchanges"]
        assert exchange["question"] == QUESTION
        assert exchange["findings"][0]["text"] == NO_LABEL
        assert float(exchange["cost_usd"]) == 0
        assert exchange["steps"] == ["Found 2 cartons", "Looked closer at 2 cartons"]
        assert len(model.requests) == paid

        await client.post(f"/api/sessions/{tapped}/questions", json={"question": "Anything else?"})
        await _settled(client, tapped)
        assert len(model.requests) == paid + 1


# A box dragged on a frame: SAM follows it from that box, the agent answers from the track, the
# proof draws its path, and the same box on the same footage is free.
async def test_a_dragged_box_is_followed(tmp_path: Path) -> None:
    sam = FollowingSam()
    flow = Flow(tmp_path, sam)
    followed = "The marked thing is in view from 00:02 to 00:06."
    model = ScriptedModel([
        completion(json.dumps(FIRST_LOOK)),
        completion("", _call("follow", {})),
        completion("", _call("answer", {"claims": [{"text": followed, "evidence": ["e4"]}]})),
    ])  # fmt: skip
    body = {"question": "Follow this", "box": BOX, "at_s": 3.0}
    async with serving(flow, model) as client:
        session_id = await _upload(client, "belt.mp4")
        await _settled(client, session_id)
        await client.post(f"/api/sessions/{session_id}/questions", json=body)
        live = await _settled(client, session_id)
        assert live["problem"] is None
        [exchange] = (await client.get(f"/api/sessions/{session_id}")).json()["exchanges"]
        assert exchange["marked_at_s"] == 3.0
        assert exchange["findings"][0]["text"] == followed
        assert exchange["findings"][0]["kind"] == "measured"
        assert (exchange["findings"][0]["start_s"], exchange["findings"][0]["end_s"]) == (2.0, 6.0)
        assert sam.prompts == ["marked thing"]
        assert "follow" in [tool.name for tool in model.requests[1].tools]
        [mark] = flow.media.drawn[-1]
        assert (mark.kind, mark.tag, len(mark.path)) == ("flagged", "followed", 9)

        asked = len(model.requests)
        same_clip = await _upload(client, "belt-again.mp4")
        await _settled(client, same_clip)
        await client.post(f"/api/sessions/{same_clip}/questions", json=body)
        await _settled(client, same_clip)
        assert len(model.requests) == asked
        assert sam.calls == 1


# An iPhone photo arrives as HEIC, stored sideways with a rotation tag. It is read upright, and a
# box dragged on it limits the count to that part.
async def test_an_iphone_photo_with_a_marked_part(tmp_path: Path) -> None:
    register_heif_opener()
    exif = Image.Exif()
    exif[0x0112] = 6
    pixels = np.zeros((200, 300, 3), np.uint8)
    pixels[:, :150] = (220, 30, 30)
    sideways = tmp_path / "IMG_4021.HEIC"
    Image.fromarray(pixels).save(sideways, format="HEIF", exif=exif.tobytes())
    flow = Flow(tmp_path / "blobs", FakeImageSam())
    flow.media.upright = PillowPhotos().upright  # type: ignore[method-assign]
    counted = "1 box in the part you marked."
    model = ScriptedModel([
        completion(json.dumps(PHOTO_LOOK)),
        completion("", _call("measure", {"thing": "box", "area": "the part you marked"})),
        completion("", _call("answer", {"claims": [{"text": counted, "evidence": ["e4"]}]})),
    ])  # fmt: skip
    body = {"question": "How many boxes here?", "box": {"x0": 0, "y0": 0, "x1": 0.3, "y1": 1}}
    async with serving(flow, model) as client:
        session_id = await _upload(client, sideways.name, sideways.read_bytes())
        await _settled(client, session_id)
        view = (await client.get(f"/api/sessions/{session_id}")).json()
        assert view["photo"]
        upright = await client.get(view["media_url"])
        assert upright.headers["content-type"] == "image/jpeg"
        saved = await flow.sessions.get(session_id)
        assert saved is not None
        assert (saved.footage.width, saved.footage.height) == (200, 300)

        await client.post(f"/api/sessions/{session_id}/questions", json=body)
        await _settled(client, session_id)
        [exchange] = (await client.get(f"/api/sessions/{session_id}")).json()["exchanges"]
        assert exchange["findings"][0]["text"] == counted
        assert exchange["findings"][0]["proof_photo"]
        assert "follow" not in [tool.name for tool in model.requests[1].tools]
        [[area, counted_box]] = flow.media.photos
        assert (area.kind, area.tag) == ("area", "the part you marked")
        assert counted_box.path[0].box.x0 == 0.1


# What cannot be read or paid for ends in one plain line, never an endless wait.
async def test_failures_end_in_a_plain_line(tmp_path: Path) -> None:
    flow = Flow(tmp_path, FakeSam())
    flow.media.unreadable = True
    async with serving(flow, ScriptedModel([])) as client:
        session_id = await _upload(client, "broken.mov")
        live = await _settled(client, session_id)
        assert "couldn't be read" in str(live["problem"])
        assert (await client.get("/api/sessions/nothing-here/live")).status_code == 404
        assert (
            await client.post("/api/sessions", files={"file": ("a.mp4", b"")})
        ).status_code == 400

    flow = Flow(tmp_path / "second", FakeSam())
    spent = SpentModel([])
    async with serving(flow, spent) as client:
        session_id = await _upload(client, "belt.mp4")
        live = await _settled(client, session_id)
        assert "budget" in str(live["problem"])
