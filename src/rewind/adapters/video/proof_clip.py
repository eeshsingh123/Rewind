import asyncio
import subprocess
from bisect import bisect_left
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from rewind.contracts import Box, TimeSpan
from rewind.core.analysis import Mark, Sighting
from rewind.core.analysis.proof import PART_SHOWN_S
from rewind.core.ports import VideoToolsError

# Colours are BGR. Red for what broke the rule, green for what passed or was found.
_RED = (40, 40, 225)
_GREEN = (70, 190, 60)
_AMBER = (30, 170, 240)
_WHITE = (255, 255, 255)
_SHADE = (20, 20, 20)
_MAX_HEIGHT = 720
# Between two sightings further apart than this the item was not seen, so nothing is drawn.
_MAX_GAP_S = 1.2
# Past its first or last sighting, an item is held in place this long.
_HOLD_S = 0.3
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_COLOURS = {"passed": _GREEN, "area": _AMBER}
_PHOTO_QUALITY = 90


class OpenCvProofClips:
    def __init__(self, ffmpeg: str = "ffmpeg") -> None:
        self._ffmpeg = ffmpeg

    async def make_clip(
        self, source: Path, destination: Path, span: TimeSpan, marks: Sequence[Mark]
    ) -> None:
        await asyncio.to_thread(_render, self._ffmpeg, source, destination, span, tuple(marks))

    async def make_photo(self, source: Path, destination: Path, marks: Sequence[Mark]) -> None:
        await asyncio.to_thread(_render_photo, source, destination, tuple(marks))


def _render(
    ffmpeg: str, source: Path, destination: Path, span: TimeSpan, marks: tuple[Mark, ...]
) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise VideoToolsError(f"cannot open {source.name}")
    encoder: subprocess.Popen[bytes] | None = None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        # Seek a little early: a seek lands on a keyframe, and frames are then read forward.
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, span.start - 2.0) * 1000)
        while True:
            t = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            ok, frame = capture.read()
            if not ok or t > span.end:
                break
            if t < span.start:
                continue
            frame = _fit(frame)
            _draw(frame, t, marks)
            if encoder is None:
                encoder = _encoder(ffmpeg, destination, frame.shape[1], frame.shape[0], fps)
            assert encoder.stdin is not None
            encoder.stdin.write(frame.tobytes())
    except BaseException:
        if encoder is not None:
            encoder.kill()
        raise
    finally:
        capture.release()
    if encoder is None:
        raise VideoToolsError(f"no frames between {span.start:.1f} s and {span.end:.1f} s")
    _finish(encoder)


def _render_photo(source: Path, destination: Path, marks: tuple[Mark, ...]) -> None:
    frame = cv2.imread(str(source))
    if frame is None:
        raise VideoToolsError(f"cannot open {source.name}")
    _draw_marks(frame, 0.0, marks)
    if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, _PHOTO_QUALITY]):
        raise VideoToolsError("could not write the proof photo")


def _fit(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    if height <= _MAX_HEIGHT:
        return frame
    # Even sizes: H.264 with yuv420p needs them.
    new_width = round(width * _MAX_HEIGHT / height / 2) * 2
    return cv2.resize(frame, (new_width, _MAX_HEIGHT), interpolation=cv2.INTER_AREA)


def _draw(frame: np.ndarray, t: float, marks: tuple[Mark, ...]) -> None:
    _draw_marks(frame, t, marks)
    scale = frame.shape[0] / 720
    _label(frame, _clock(t), (round(12 * scale), round(40 * scale)), _SHADE, scale)


def _draw_marks(frame: np.ndarray, t: float, marks: tuple[Mark, ...]) -> None:
    height, width = frame.shape[:2]
    scale = height / 720
    # Flagged last, so its box and tag sit on top.
    for mark in sorted(marks, key=lambda m: m.kind == "flagged"):
        box = mark.path[0].box if mark.kind == "area" else _box_at(mark.path, t)
        if box is None:
            continue
        flagged = mark.kind == "flagged"
        colour = _COLOURS.get(mark.kind, _RED)
        x0, y0, x1, y1 = _pixels(box, width, height)
        cv2.rectangle(
            frame, (x0, y0), (x1, y1), colour, max(2, round((5 if flagged else 2) * scale))
        )
        part = _part_at(mark.parts, t)
        if part is not None:
            px0, py0, px1, py1 = _pixels(_inside(part, box), width, height)
            cv2.rectangle(frame, (px0, py0), (px1, py1), _GREEN, max(2, round(2 * scale)))
        if mark.tag:
            _label(frame, mark.tag, (x0, y0), colour, scale)


def _box_at(path: Sequence[Sighting], t: float) -> Box | None:
    index = bisect_left(path, t, key=lambda s: s.t)
    before = path[index - 1] if index > 0 else None
    after = path[index] if index < len(path) else None
    if before is not None and after is not None:
        if after.t - before.t > _MAX_GAP_S:
            return None
        share = (t - before.t) / (after.t - before.t) if after.t > before.t else 0.0
        return _lerp(before.box, after.box, share)
    nearest = before or after
    if nearest is None or abs(nearest.t - t) > _HOLD_S:
        return None
    return nearest.box


def _part_at(parts: Sequence[Sighting], t: float) -> Box | None:
    near = [p for p in parts if abs(p.t - t) <= PART_SHOWN_S]
    return min(near, key=lambda p: abs(p.t - t)).box if near else None


def _lerp(a: Box, b: Box, share: float) -> Box:
    def mix(p: float, q: float) -> float:
        return p + (q - p) * share

    return Box(x0=mix(a.x0, b.x0), y0=mix(a.y0, b.y0), x1=mix(a.x1, b.x1), y1=mix(a.y1, b.y1))


# A part's box is kept relative to its item's box, so it moves with the item between sightings.
def _inside(part: Box, whole: Box) -> Box:
    width, height = whole.x1 - whole.x0, whole.y1 - whole.y0
    return Box(
        x0=whole.x0 + part.x0 * width, y0=whole.y0 + part.y0 * height,
        x1=whole.x0 + part.x1 * width, y1=whole.y0 + part.y1 * height,
    )  # fmt: skip


def _pixels(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    return (
        round(box.x0 * (width - 1)), round(box.y0 * (height - 1)),
        round(box.x1 * (width - 1)), round(box.y1 * (height - 1)),
    )  # fmt: skip


# Text on a filled band above the point, or below it when the point is at the top of the frame.
def _label(
    frame: np.ndarray, text: str, at: tuple[int, int], colour: tuple[int, int, int], scale: float
) -> None:
    size, thickness = 0.8 * scale, max(1, round(2 * scale))
    (text_width, text_height), baseline = cv2.getTextSize(text, _FONT, size, thickness)
    pad = round(6 * scale)
    x = min(max(0, at[0]), frame.shape[1] - text_width - 2 * pad)
    top = at[1] - text_height - baseline - 2 * pad
    if top < 0:
        top = at[1]
    bottom = top + text_height + baseline + 2 * pad
    cv2.rectangle(frame, (x, top), (x + text_width + 2 * pad, bottom), colour, cv2.FILLED)
    cv2.putText(
        frame, text, (x + pad, bottom - pad - baseline), _FONT, size, _WHITE, thickness,
        cv2.LINE_AA,
    )  # fmt: skip


def _clock(t: float) -> str:
    minutes, seconds = divmod(int(t), 60)
    return f"{minutes:02d}:{seconds:02d}"


def _encoder(
    ffmpeg: str, destination: Path, width: int, height: int, fps: float
) -> subprocess.Popen[bytes]:
    command = [
        ffmpeg, "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:g}",
        "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(destination),
    ]  # fmt: skip
    try:
        return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as error:
        raise VideoToolsError(f"{ffmpeg} is not installed or not on PATH") from error


def _finish(encoder: subprocess.Popen[bytes]) -> None:
    _, stderr = encoder.communicate()
    if encoder.returncode != 0:
        tail = stderr.decode(errors="replace").strip().splitlines()[-3:]
        raise VideoToolsError(f"ffmpeg failed: {' | '.join(tail)}")
