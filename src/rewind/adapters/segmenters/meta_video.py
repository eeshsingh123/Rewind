import asyncio
import logging
from decimal import Decimal

from rewind.adapters.meta_files import MetaFiles, MetaFilesError
from rewind.adapters.segmenters.meta_api import MetaApi
from rewind.adapters.segmenters.sam_wire import (
    MODEL,
    billed_frames,
    parse_observations,
    request_body,
)
from rewind.contracts import Charge
from rewind.core.ports import Segmentation, SegmenterError, SegmentRequest

USD_PER_FRAME = Decimal("0.20") / 1000
log = logging.getLogger(__name__)


class MetaVideoSegmenter:
    def __init__(self, api: MetaApi, files: MetaFiles) -> None:
        self._api = api
        self._files = files

    @property
    def model_version(self) -> str:
        return MODEL

    @property
    def max_frames(self) -> int:
        return 15_000

    @property
    def max_prompt_objects(self) -> int:
        return 16

    # Seen in the API checks and on the stock clip: after 16 object ids, no new objects are found.
    @property
    def max_tracked_objects(self) -> int | None:
        return 16

    @property
    def min_billed_frames(self) -> int:
        return 50

    def estimate(self, request: SegmentRequest) -> tuple[Charge, ...]:
        return (_charge(billed_frames(request, self.min_billed_frames)),)

    async def segment(self, request: SegmentRequest) -> Segmentation:
        try:
            file_id = await self._files.file_id(request.media_key)
        except MetaFilesError as error:
            raise SegmenterError(str(error)) from error
        body = request_body(request, {"type": "input_video", "file_id": file_id})
        response = await self._api.stream_response(body)
        frames = response.usage.video_frames_processed if response.usage else None
        if frames is None:
            frames = billed_frames(request, self.min_billed_frames)
            log.warning("SAM reported no video_frames_processed; recording %d estimated", frames)
        observations = await asyncio.to_thread(parse_observations, response.text, request)
        return Segmentation(observations=observations, charges=(_charge(frames),))


def _charge(frames: int) -> Charge:
    return Charge(
        provider="meta",
        model=MODEL,
        unit="video_frame",
        quantity=frames,
        usd=frames * USD_PER_FRAME,
    )
