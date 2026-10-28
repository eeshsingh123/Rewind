import asyncio
import logging
import time
from decimal import Decimal

from rewind.adapters.segmenters.meta_api import MetaApi
from rewind.adapters.segmenters.sam_wire import (
    MODEL,
    billed_frames,
    parse_observations,
    request_body,
)
from rewind.contracts import Charge
from rewind.core.ports import BlobStore, Segmentation, SegmentRequest

USD_PER_FRAME = Decimal("0.20") / 1000
# An uploaded file is reused for every concept only while it has at least this long to live.
_REUSE_MARGIN_S = 600
log = logging.getLogger(__name__)


class MetaVideoSegmenter:
    def __init__(self, api: MetaApi, blobs: BlobStore, file_ttl_s: int = 86_400) -> None:
        self._api = api
        self._blobs = blobs
        self._file_ttl_s = file_ttl_s
        self._uploads: dict[str, tuple[str, float]] = {}

    @property
    def model_version(self) -> str:
        return MODEL

    @property
    def max_frames(self) -> int:
        return 15_000

    @property
    def max_prompt_objects(self) -> int:
        return 16

    # S0 and the stock conveyor clip: after 16 object ids, a request finds no new objects.
    @property
    def max_tracked_objects(self) -> int | None:
        return 16

    @property
    def min_billed_frames(self) -> int:
        return 50

    def estimate(self, request: SegmentRequest) -> tuple[Charge, ...]:
        return (_charge(billed_frames(request, self.min_billed_frames)),)

    async def segment(self, request: SegmentRequest) -> Segmentation:
        file_id = await self._file_id(request.media_key)
        body = request_body(request, {"type": "input_video", "file_id": file_id})
        response = await self._api.stream_response(body)
        frames = response.usage.video_frames_processed if response.usage else None
        if frames is None:
            frames = billed_frames(request, self.min_billed_frames)
            log.warning("SAM reported no video_frames_processed; recording %d estimated", frames)
        observations = await asyncio.to_thread(parse_observations, response.text, request)
        return Segmentation(observations=observations, charges=(_charge(frames),))

    async def _file_id(self, media_key: str) -> str:
        cached = self._uploads.get(media_key)
        now = time.monotonic()
        if cached is not None and cached[1] - now > _REUSE_MARGIN_S:
            return cached[0]
        path = await self._blobs.local_path(media_key)
        file_id = await self._api.upload_file(path, "video/mp4", self._file_ttl_s)
        self._uploads[media_key] = (file_id, now + self._file_ttl_s)
        return file_id


def _charge(frames: int) -> Charge:
    return Charge(
        provider="meta",
        model=MODEL,
        unit="video_frame",
        quantity=frames,
        usd=frames * USD_PER_FRAME,
    )
