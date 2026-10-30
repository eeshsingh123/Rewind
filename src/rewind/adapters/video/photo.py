import asyncio
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from rewind.core.ports import VideoToolsError

# A phone photo is far bigger than either model reads; this keeps small items and bounds the upload.
_MAX_SIDE = 2048
_QUALITY = 92

# iPhones save HEIC by default, which OpenCV and ffmpeg's HEIF support both fail to read.
register_heif_opener()


class PillowPhotos:
    async def upright(self, source: Path, destination: Path) -> tuple[int, int]:
        return await asyncio.to_thread(_upright, source, destination)


# Applies the camera's rotation, so every box from here on fits what a person sees.
def _upright(source: Path, destination: Path) -> tuple[int, int]:
    try:
        with Image.open(source) as opened:
            photo = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise VideoToolsError(f"cannot read {source.name} as a photo") from error
    photo.thumbnail((_MAX_SIDE, _MAX_SIDE), Image.Resampling.LANCZOS)
    try:
        photo.save(destination, "JPEG", quality=_QUALITY)
    except OSError as error:
        raise VideoToolsError("could not write the photo") from error
    return photo.size
