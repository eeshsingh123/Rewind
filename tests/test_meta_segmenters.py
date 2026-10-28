import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from rewind.adapters.segmenters.meta_api import MetaApi
from rewind.adapters.segmenters.meta_image import MetaImageSegmenter
from rewind.adapters.segmenters.meta_video import MetaVideoSegmenter
from rewind.adapters.storage.local_disk import LocalDiskBlobStore
from rewind.contracts import Box
from rewind.core.ports import (
    ConceptPrompt,
    ObjectPrompt,
    ObjectPrompts,
    Prompt,
    SegmenterError,
    SegmentRequest,
)

pytestmark = pytest.mark.anyio

RECORDINGS = Path(__file__).parent / "recordings" / "segmenter"


def _recording(name: str) -> dict[str, Any]:
    return json.loads((RECORDINGS / name).read_text(encoding="utf-8"))


# Serves recorded Meta API responses and keeps the requests the adapter sent.
class Replay:
    def __init__(self, upload: str | None, *responses: str) -> None:
        self._upload = upload
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/files"):
            assert self._upload is not None
            return httpx.Response(200, json=_recording(self._upload)["body"])
        body = _recording(self._responses.pop(0))["body"]
        if "sse_events" not in body:
            return httpx.Response(200, json=body)
        lines = [
            f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n" for e in body["sse_events"]
        ]
        return httpx.Response(
            200, text="".join(lines), headers={"content-type": "text/event-stream"}
        )

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.url.path.endswith("/responses")]


async def _blobs(tmp_path: Path) -> LocalDiskBlobStore:
    blobs = LocalDiskBlobStore(tmp_path, "http://localhost", b"test")
    await blobs.write("proxy.mp4", b"not a real video")
    await blobs.write("frame.jpg", b"not a real image")
    return blobs


def _api(replay: Replay) -> MetaApi:
    client = httpx.AsyncClient(transport=httpx.MockTransport(replay))
    return MetaApi(client, "test-key", "https://api.meta.ai/v1")


def _request(frames: int, prompt: Prompt, media_key: str = "proxy.mp4") -> SegmentRequest:
    return SegmentRequest(
        footage_id="f1",
        media_key=media_key,
        width=1280,
        height=720,
        frame_times=tuple(10.0 + i * 0.5 for i in range(frames)),
        prompt=prompt,
    )


async def test_video_concept_maps_frames_to_footage_time_and_bills_reported_usage(
    tmp_path: Path,
) -> None:
    replay = Replay("t2_01_files_upload.json", "t2_02_video_rectangle_2fps.json")
    segmenter = MetaVideoSegmenter(_api(replay), await _blobs(tmp_path), file_ttl_s=3600)

    result = await segmenter.segment(_request(24, ConceptPrompt("rectangle")))

    first = result.observations[0]
    # The first record is "<2f>0<|box;x1=1118;y1=284;x2=1279;y2=433;w=1280;h=720...".
    assert (first.t, first.object_id, first.concept) == (11.0, 0, "rectangle")
    assert first.box == Box(x0=1118 / 1280, y0=284 / 720, x1=1.0, y1=434 / 720)
    assert first.mask is not None
    assert (first.mask.height, first.mask.width) == (150, 162)
    assert [(c.unit, c.quantity, c.usd) for c in result.charges] == [
        ("video_frame", 50, Decimal("0.0100"))
    ]
    body = replay.bodies()[0]
    assert body["stream"] is True
    assert "background" not in body
    assert body["input"][0]["content"][1] == {
        "type": "input_video",
        "file_id": _recording("t2_01_files_upload.json")["body"]["id"],
    }
    upload = replay.requests[0].content.decode(errors="replace")
    assert 'name="expires_after[seconds]"\r\n\r\n3600' in upload


async def test_video_uploads_a_clip_once_for_every_concept(tmp_path: Path) -> None:
    replay = Replay(
        "t2_01_files_upload.json",
        "t2_02_video_rectangle_2fps.json",
        "t2_03_video_zero_matches.json",
    )
    segmenter = MetaVideoSegmenter(_api(replay), await _blobs(tmp_path))

    await segmenter.segment(_request(24, ConceptPrompt("rectangle")))
    empty = await segmenter.segment(_request(24, ConceptPrompt("brown box")))

    assert sum(r.url.path.endswith("/files") for r in replay.requests) == 1
    assert empty.observations == ()
    assert empty.charges[0].quantity == 50


async def test_video_short_clip_is_billed_the_50_frame_floor_as_estimated(tmp_path: Path) -> None:
    replay = Replay("05_files_upload.json", "07b_sam_video_10frames_stream.json")
    segmenter = MetaVideoSegmenter(_api(replay), await _blobs(tmp_path))
    request = _request(10, ConceptPrompt("rectangle"))

    result = await segmenter.segment(request)

    assert segmenter.estimate(request) == result.charges
    assert result.charges[0].quantity == 50


async def test_video_box_prompt_tracks_forward_under_our_object_id(tmp_path: Path) -> None:
    replay = Replay("t2_01_files_upload.json", "t2_04_video_box_forward.json")
    segmenter = MetaVideoSegmenter(_api(replay), await _blobs(tmp_path))
    box = Box(x0=0.40625, y0=0.3958, x1=0.5625, y1=0.6042)
    prompt = ObjectPrompts(
        concept="carton",
        objects=(ObjectPrompt(t=18.0, object_id=1, target=box),),
        direction="forward",
    )

    result = await segmenter.segment(_request(24, prompt))

    body = replay.bodies()[0]
    assert body["input"][0]["content"][0]["text"] == (
        "<16f>1<|box;x1=520;y1=284;x2=719;y2=435;w=1280;h=720|>"
    )
    assert body["metadata"]["propagation_direction"] == "forward"
    assert {o.object_id for o in result.observations} == {1}
    assert {o.concept for o in result.observations} == {"carton"}
    assert min(o.t for o in result.observations) == 18.0


async def test_image_reads_records_and_bills_one_image(tmp_path: Path) -> None:
    replay = Replay(None, "02_sam_image_brown_box.json")
    segmenter = MetaImageSegmenter(_api(replay), await _blobs(tmp_path))
    request = SegmentRequest(
        footage_id="f1",
        media_key="frame.jpg",
        width=1280,
        height=720,
        frame_times=(3.0,),
        prompt=ConceptPrompt("brown box"),
    )

    result = await segmenter.segment(request)

    assert [(o.t, o.object_id, round(o.score, 2)) for o in result.observations] == [
        (3.0, 0, 0.51),
        (3.0, 1, 0.52),
    ]
    assert result.charges[0].unit == "image"
    assert result.charges[0].usd == Decimal("0.0025")
    assert replay.bodies()[0]["input"][0]["content"][1]["image_url"].startswith("data:image/jpeg")


async def test_a_rejected_request_raises_the_port_error(tmp_path: Path) -> None:
    def reject(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad prompt"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(reject))
    segmenter = MetaVideoSegmenter(
        MetaApi(client, "k", "https://api.meta.ai/v1"), await _blobs(tmp_path)
    )

    with pytest.raises(SegmenterError, match="400"):
        await segmenter.segment(_request(24, ConceptPrompt("rectangle")))
