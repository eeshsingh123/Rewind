import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from rewind.adapters.meta_files import MetaFiles
from rewind.adapters.segmenters.meta_api import MetaApi
from rewind.adapters.segmenters.meta_video import MetaVideoSegmenter
from rewind.adapters.storage.local_disk import LocalDiskBlobStore
from rewind.contracts import Box
from rewind.core.ports import (
    ConceptPrompt,
    Prompt,
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
    return blobs


def _client(replay: Replay) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(replay))


def _api(replay: Replay) -> MetaApi:
    return MetaApi(_client(replay), "test-key", "https://api.meta.ai/v1")


def _request(frames: int, prompt: Prompt) -> SegmentRequest:
    return SegmentRequest(
        footage_id="f1",
        media_key="proxy.mp4",
        width=1280,
        height=720,
        frame_times=tuple(10.0 + i * 0.5 for i in range(frames)),
        prompt=prompt,
    )


async def test_video_concept_maps_frames_to_footage_time_and_bills_reported_usage(
    tmp_path: Path,
) -> None:
    replay = Replay("t2_01_files_upload.json", "t2_02_video_rectangle_2fps.json")
    files = MetaFiles(_client(replay), "test-key", "https://api.meta.ai/v1",
                      await _blobs(tmp_path), ttl_s=3600)  # fmt: skip
    segmenter = MetaVideoSegmenter(_api(replay), files)

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
