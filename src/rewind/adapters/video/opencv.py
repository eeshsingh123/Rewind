import asyncio
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from rewind.contracts import Box
from rewind.core.perception.activity import ActivitySample
from rewind.core.ports import VideoToolsError

# Motion is measured on a small, blurred frame: cheaper, and it smooths away sensor noise.
_WORK_WIDTH = 320
_BLUR = (5, 5)
_PIXEL_CHANGE = 25
_OPEN_KERNEL = np.ones((3, 3), np.uint8)


# Differences between sampled frames, not MOG2: a background model learns a belt that is
# always busy as background and reports it idle, and skipped footage is a missed violation.
class OpenCvActivityProbe:
    async def sample(
        self, source: Path, sample_fps: float, regions: Sequence[Box]
    ) -> tuple[ActivitySample, ...]:
        return await asyncio.to_thread(_sample, source, sample_fps, tuple(regions))


def _sample(
    source: Path, sample_fps: float, regions: tuple[Box, ...]
) -> tuple[ActivitySample, ...]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise VideoToolsError(f"cannot open {source.name}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or sample_fps
        step = max(1, round(fps / sample_fps))
        samples: list[ActivitySample] = []
        previous: np.ndarray | None = None
        index = 0
        while capture.grab():
            if index % step == 0:
                ok, frame = capture.retrieve()
                if not ok:
                    break
                current = _prepare(frame)
                if previous is not None:
                    samples.append(
                        ActivitySample(t=index / fps, score=_score(previous, current, regions))
                    )
                previous = current
            index += 1
        return tuple(samples)
    finally:
        capture.release()


def _prepare(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    small = cv2.resize(frame, (_WORK_WIDTH, max(1, round(height * _WORK_WIDTH / width))))
    return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), _BLUR, 0)


def _score(previous: np.ndarray, current: np.ndarray, regions: tuple[Box, ...]) -> float:
    changed = (cv2.absdiff(previous, current) > _PIXEL_CHANGE).astype(np.uint8)
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, _OPEN_KERNEL)
    rows, cols = changed.shape[:2]
    scores = [0.0]
    for box in regions:
        crop = changed[
            int(box.y0 * rows) : max(int(box.y0 * rows) + 1, int(box.y1 * rows)),
            int(box.x0 * cols) : max(int(box.x0 * cols) + 1, int(box.x1 * cols)),
        ]
        scores.append(float(np.count_nonzero(crop)) / crop.size)
    return max(scores)
