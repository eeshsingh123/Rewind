import asyncio
from pathlib import Path

import cv2

from rewind.core.ports import VideoToolsError

# A phone photo is far bigger than either model reads; this keeps small items and bounds the upload.
_MAX_SIDE = 2048
_QUALITY = 92


class OpenCvPhotos:
    async def upright(self, source: Path, destination: Path) -> tuple[int, int]:
        return await asyncio.to_thread(_upright, source, destination)


# imread applies the camera's rotation, so every box from here on fits what a person sees.
def _upright(source: Path, destination: Path) -> tuple[int, int]:
    photo = cv2.imread(str(source))
    if photo is None:
        raise VideoToolsError(f"cannot read {source.name} as a photo")
    height, width = photo.shape[:2]
    scale = _MAX_SIDE / max(width, height)
    if scale < 1:
        width, height = round(width * scale), round(height * scale)
        photo = cv2.resize(photo, (width, height), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(destination), photo, [cv2.IMWRITE_JPEG_QUALITY, _QUALITY]):
        raise VideoToolsError("could not write the photo")
    return width, height
