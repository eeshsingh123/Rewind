import logging
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
from meta_sam_parser import (
    FramePrompt,
    InvalidSegmentationMaskError,
    PromptBox,
    PromptObject,
    PromptPoint,
    SegmentationMask,
    SegmentationMaskEncoding,
    build_object_prompt,
    decode_mask_to_raster,
)

from rewind.contracts import Box, Observation, RleMask
from rewind.core.ports import ConceptPrompt, ObjectPrompts, Prompt, SegmentRequest

MODEL = "sam-3.1"
# One line per frame: "<12f>" then comma-separated records. A mask payload's alphabet excludes
# "|", ";", ",", "<" and ">", so these patterns cannot cut a record short.
_LINE = re.compile(r"<(\d+)f>([^\n]*)")
_RECORD = re.compile(r"(\d+)<\|box;([^|]*)\|><\|mask;([^|]*)\|>")
_ENCODINGS: dict[str, SegmentationMaskEncoding] = {"~": "lossless", "!": "one_bit"}
log = logging.getLogger(__name__)


def request_body(request: SegmentRequest, media: dict[str, str]) -> dict[str, Any]:
    metadata = {"include_confidence": "true", "mask_encoding": "one_bit"}
    if isinstance(request.prompt, ObjectPrompts):
        metadata["propagation_direction"] = request.prompt.direction
    content = [{"type": "input_text", "text": prompt_text(request)}, media]
    return {
        "model": MODEL,
        "input": [{"type": "message", "role": "user", "content": content}],
        "metadata": metadata,
    }


def prompt_text(request: SegmentRequest) -> str:
    prompt = request.prompt
    if isinstance(prompt, ConceptPrompt):
        return prompt.text
    width, height = request.width, request.height
    frames: defaultdict[int, list[FramePrompt]] = defaultdict(list)
    for marked in prompt.objects:
        frame = nearest_frame(request.frame_times, marked.t)
        if isinstance(marked.target, Box):
            box = PromptBox(
                left=math.floor(marked.target.x0 * width),
                top=math.floor(marked.target.y0 * height),
                right=math.ceil(marked.target.x1 * width),
                bottom=math.ceil(marked.target.y1 * height),
            )
            frames[marked.object_id].append(FramePrompt(frame=frame, box=box))
        else:
            points = [
                PromptPoint(
                    x=min(width - 1, int(p.point.x * width)),
                    y=min(height - 1, int(p.point.y * height)),
                    label="positive" if p.positive else "negative",
                )
                for p in marked.target
            ]
            frames[marked.object_id].append(FramePrompt(frame=frame, points=points))
    objects = [PromptObject(id=object_id, prompts=p) for object_id, p in frames.items()]
    return build_object_prompt(size=(width, height), objects=objects)


def nearest_frame(frame_times: Sequence[float], t: float) -> int:
    return min(range(len(frame_times)), key=lambda index: abs(frame_times[index] - t))


def concept_of(prompt: Prompt) -> str:
    return prompt.text if isinstance(prompt, ConceptPrompt) else prompt.concept


def billed_frames(request: SegmentRequest, minimum: int) -> int:
    frames = len(request.frame_times)
    prompt = request.prompt
    if isinstance(prompt, ObjectPrompts) and prompt.direction != "both":
        first = min(nearest_frame(request.frame_times, marked.t) for marked in prompt.objects)
        frames = frames - first if prompt.direction == "forward" else first + 1
    return max(minimum, frames)


# The call is already billed by now, so a malformed record is logged and skipped, never fatal.
# Records are read by the grammar rather than meta-sam-parser's stream parser, which decodes every
# mask to validate it: mask decoding is pure Python and costs about 0.1 s per mask.
def parse_observations(text: str, request: SegmentRequest) -> tuple[Observation, ...]:
    observations: list[Observation] = []
    for line in _LINE.finditer(text):
        frame = int(line.group(1))
        if frame >= len(request.frame_times):
            log.warning("SAM returned frame %d of a %d-frame clip", frame, len(request.frame_times))
            continue
        for record in _RECORD.finditer(line.group(2)):
            try:
                observations.append(_observation(request, frame, record))
            except (KeyError, ValueError, InvalidSegmentationMaskError) as error:
                log.warning("skipped a malformed SAM record on frame %d: %s", frame, error)
    return tuple(observations)


def _observation(request: SegmentRequest, frame: int, record: re.Match[str]) -> Observation:
    box = _fields(record.group(2))
    mask = _fields(record.group(3))
    width, height = request.width, request.height
    # x2 and y2 are inclusive pixels; the box we keep is half-open.
    return Observation(
        footage_id=request.footage_id,
        t=request.frame_times[frame],
        concept=concept_of(request.prompt),
        box=Box(
            x0=_unit(int(box["x1"]) / width),
            y0=_unit(int(box["y1"]) / height),
            x1=_unit((int(box["x2"]) + 1) / width),
            y1=_unit((int(box["y2"]) + 1) / height),
        ),
        score=_unit(float(box.get("c", 1.0))),
        object_id=int(record.group(1)),
        mask=_mask(mask["data"]),
    )


def _fields(token: str) -> dict[str, str]:
    return dict(field.split("=", 1) for field in token.split(";") if "=" in field)


def _mask(data: str) -> RleMask:
    height, width, payload = data.split(",", 2)
    encoding = _ENCODINGS[payload[0]]
    mask = SegmentationMask(
        encoding=encoding, payload=payload, width=int(width), height=int(height)
    )
    values = np.frombuffer(decode_mask_to_raster(mask), dtype=np.uint8) != 0
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    runs = np.diff(np.concatenate(([0], changes, [values.size]))).tolist()
    # Runs start with 0s, so a mask whose first pixel is set opens with an empty run.
    if values.size and values[0]:
        runs = [0, *runs]
    return RleMask(height=int(height), width=int(width), counts=tuple(runs))


def _unit(value: float) -> float:
    return min(1.0, max(0.0, value))
