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


class OpenCvContactSheet:
    async def make_sheet(self, source: Path, destination: Path, tiles: Sequence[Tile]) -> None:
        await asyncio.to_thread(_sheet, source, destination, tuple(tiles))


def _sheet(source: Path, destination: Path, tiles: tuple[Tile, ...]) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise VideoToolsError(f"cannot open {source.name}")
    try:
        images = [_tile(capture, tile) for tile in tiles]
    finally:
        capture.release()
    rows = -(-len(images) // _COLUMNS)
    columns = min(_COLUMNS, len(images))
    sheet = np.full((rows * _TILE_H, columns * _TILE_W, 3), 255, dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, _COLUMNS)
        sheet[row * _TILE_H : (row + 1) * _TILE_H, column * _TILE_W : (column + 1) * _TILE_W] = (
            image
        )
    if not cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85]):
        raise VideoToolsError("could not write the insight picture")


def _tile(capture: cv2.VideoCapture, tile: Tile) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_MSEC, tile.t * 1000)
    ok, frame = capture.read()
    if not ok:
        raise VideoToolsError(f"no frame at {tile.t:.1f} s")
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
    image = cv2.resize(frame[y0:y1, x0:x1], (_TILE_W, _TILE_H), interpolation=cv2.INTER_AREA)
    cv2.rectangle(image, (0, 0), (_TILE_W, 26), _SHADE, -1)
    cv2.putText(image, tile.label, (6, 19), _FONT, 0.6, _WHITE, 1, cv2.LINE_AA)
    return image
