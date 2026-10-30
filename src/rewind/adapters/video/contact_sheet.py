import asyncio
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from rewind.core.analysis.facts import Tile
from rewind.core.ports import VideoToolsError

# BGR, as in the proof clips.
_RED = (40, 40, 225)
_WHITE = (255, 255, 255)
_SHADE = (20, 20, 20)
_TILE_W, _TILE_H = 400, 300
_COLUMNS = 3
# A tile shows the item with this much of its surroundings, and never less than this share of
# the frame, so the model sees what the item looks like and where it is.
_CONTEXT = 2.5
_MIN_SHARE = 0.35
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PHOTOS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


class OpenCvContactSheet:
    async def make_sheet(self, source: Path, destination: Path, tiles: Sequence[Tile]) -> None:
        await asyncio.to_thread(_sheet, source, destination, tuple(tiles))


def _sheet(source: Path, destination: Path, tiles: tuple[Tile, ...]) -> None:
    # A lone tile gets the width of a full sheet: a whole frame or an area needs the detail.
    grow = _COLUMNS if len(tiles) == 1 else 1
    width, height = _TILE_W * grow, _TILE_H * grow
    frames = _frames(source, tiles)
    images = [_tile(frame, tile, width, height) for frame, tile in zip(frames, tiles, strict=True)]
    rows = -(-len(images) // _COLUMNS)
    columns = min(_COLUMNS, len(images))
    sheet = np.full((rows * height, columns * width, 3), 255, dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, _COLUMNS)
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = image
    if not cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        raise VideoToolsError("could not write the insight picture")


def _frames(source: Path, tiles: tuple[Tile, ...]) -> list[np.ndarray]:
    if source.suffix.lower() in _PHOTOS:
        photo = cv2.imread(str(source))
        if photo is None:
            raise VideoToolsError(f"cannot open {source.name}")
        return [photo.copy() for _ in tiles]
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise VideoToolsError(f"cannot open {source.name}")
    try:
        return [_frame_at(capture, tile.t) for tile in tiles]
    finally:
        capture.release()


def _frame_at(capture: cv2.VideoCapture, t: float) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = capture.read()
    if not ok:
        raise VideoToolsError(f"no frame at {t:.1f} s")
    return frame


def _tile(frame: np.ndarray, tile: Tile, tile_w: int, tile_h: int) -> np.ndarray:
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = 0, 0, width, height
    if tile.box is not None:
        box = tile.box
        bx0, by0 = int(box.x0 * width), int(box.y0 * height)
        bx1, by1 = int(box.x1 * width), int(box.y1 * height)
        cv2.rectangle(frame, (bx0, by0), (bx1, by1), _RED, max(2, width // 400))
        side_w = max((bx1 - bx0) * _CONTEXT, width * _MIN_SHARE)
        side_h = max((by1 - by0) * _CONTEXT, height * _MIN_SHARE, side_w * _TILE_H / _TILE_W)
        side_w = max(side_w, side_h * _TILE_W / _TILE_H)
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        x0 = int(min(max(0, cx - side_w / 2), max(0, width - side_w)))
        y0 = int(min(max(0, cy - side_h / 2), max(0, height - side_h)))
        x1, y1 = int(min(width, x0 + side_w)), int(min(height, y0 + side_h))
    image = _letterboxed(frame[y0:y1, x0:x1], tile_w, tile_h)
    cv2.rectangle(image, (0, 0), (tile_w, 26), _SHADE, -1)
    cv2.putText(image, tile.label, (6, 19), _FONT, 0.6, _WHITE, 1, cv2.LINE_AA)
    return image


# A whole frame or a wide box rarely has the tile's shape; stretching would change what it shows.
def _letterboxed(crop: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / crop.shape[1], height / crop.shape[0])
    fitted_w = max(1, round(crop.shape[1] * scale))
    fitted_h = max(1, round(crop.shape[0] * scale))
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    left, top = (width - fitted_w) // 2, (height - fitted_h) // 2
    image[top : top + fitted_h, left : left + fitted_w] = cv2.resize(
        crop, (fitted_w, fitted_h), interpolation=cv2.INTER_AREA
    )
    return image
