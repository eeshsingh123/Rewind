import subprocess
from pathlib import Path

import cv2
import numpy as np

from rewind.contracts import Polygon
from sim.plan import ItemPlan, ScenePlan
from sim.projection import project

ZONE_COLOUR = (255, 200, 0)
OK_COLOUR = (80, 220, 80)
BAD_COLOUR = (40, 40, 255)


def draw_preview(plan: ScenePlan, video: Path, output: Path) -> None:
    # Draws the answer key over the rendered video: zones, item boxes and ids, parts, violations.
    config = plan.config
    capture = cv2.VideoCapture(str(video))
    width, height = config.width, config.height
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(config.fps), "-i", "-",
            "-c:v", "libx264", "-crf", "26", "-pix_fmt", "yuv420p", str(output),
        ],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    assert encoder.stdin is not None
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        _draw_frame(plan, frame, index / config.fps)
        encoder.stdin.write(frame.tobytes())
        index += 1
    capture.release()
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg failed to write the preview")


def item_box(plan: ScenePlan, item: ItemPlan, t: float) -> tuple[float, float, float, float] | None:
    line = plan.line
    x = item.offset + plan.belt.distance(t)
    length, width, height = line.item_size
    aspect = plan.config.width / plan.config.height
    corners = [
        project(plan.camera, (x + dx * length / 2, dy * width / 2, line.belt_top + dz * height), aspect)
        for dx in (-1, 1)
        for dy in (-1, 1)
        for dz in (0, 1)
    ]
    points = [c for c in corners if c is not None]
    if len(points) < len(corners):
        return None
    us, vs = [p[0] for p in points], [p[1] for p in points]
    return min(us), min(vs), max(us), max(vs)


def _draw_frame(plan: ScenePlan, frame: np.ndarray, t: float) -> None:
    height, width = frame.shape[:2]
    for zone in plan.zones:
        if isinstance(zone.shape, Polygon):
            points = np.array([[p.x * width, p.y * height] for p in zone.shape.points], np.int32)
            cv2.polylines(frame, [points], True, ZONE_COLOUR, 2)
            cv2.putText(frame, zone.key, tuple(points[3] + [2, -6]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ZONE_COLOUR, 1)
    for item in plan.items:
        if not item.timeline.first_seen <= t <= item.timeline.last_seen:
            continue
        box = item_box(plan, item, t)
        if box is None:
            continue
        colour = BAD_COLOUR if item.violations else OK_COLOUR
        x0, y0, x1, y1 = (int(box[0] * width), int(box[1] * height), int(box[2] * width), int(box[3] * height))
        cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2)
        name = item.id if item.order_id is None else f"{item.id} #{item.order_id}"
        parts = " ".join(f"{p}{'+' if item.timeline.has(p, t) else '-'}" for p in item.timeline.parts)
        lines = [name, parts, *item.violations]
        for row, text in enumerate(lines):
            y = y0 - 8 - 15 * (len(lines) - 1 - row)
            cv2.putText(frame, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour if row != 1 else (255, 255, 255), 1, cv2.LINE_AA)
    banner = f"t={t:5.1f}s"
    if any(jam.start <= t < jam.end for jam in plan.belt.jams):
        banner += "  JAM"
    cv2.putText(frame, banner, (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
