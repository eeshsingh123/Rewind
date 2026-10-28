import asyncio
import base64
from decimal import Decimal
from pathlib import PurePosixPath

from rewind.adapters.segmenters.meta_api import MetaApi
from rewind.adapters.segmenters.sam_wire import MODEL, parse_observations, request_body
from rewind.contracts import Charge
from rewind.core.ports import BlobStore, Segmentation, SegmenterError, SegmentRequest

USD_PER_IMAGE = Decimal("2.50") / 1000
_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


class MetaImageSegmenter:
    def __init__(self, api: MetaApi, blobs: BlobStore) -> None:
        self._api = api
        self._blobs = blobs

    @property
    def model_version(self) -> str:
        return MODEL

    @property
    def max_frames(self) -> int:
        return 1

    @property
    def max_prompt_objects(self) -> int:
        return 16

    @property
    def min_billed_frames(self) -> int:
        return 1

    def estimate(self, request: SegmentRequest) -> tuple[Charge, ...]:
        return (_charge(1),)

    async def segment(self, request: SegmentRequest) -> Segmentation:
        if len(request.frame_times) != 1:
            raise SegmenterError("an image request has exactly one frame time")
        mime = _MIME.get(PurePosixPath(request.media_key).suffix.lower())
        if mime is None:
            raise SegmenterError(f"unsupported image type: {request.media_key}")
        encoded = base64.b64encode(await self._blobs.read(request.media_key)).decode()
        media = {"type": "input_image", "image_url": f"data:{mime};base64,{encoded}"}
        response = await self._api.create_response(request_body(request, media))
        images = response.usage.images_processed if response.usage else None
        observations = await asyncio.to_thread(parse_observations, response.text, request)
        return Segmentation(
            observations=observations, charges=(_charge(images if images is not None else 1),)
        )


def _charge(images: int) -> Charge:
    return Charge(
        provider="meta", model=MODEL, unit="image", quantity=images, usd=images * USD_PER_IMAGE
    )
