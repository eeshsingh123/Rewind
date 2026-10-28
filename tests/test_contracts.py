from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from rewind.contracts import (
    Box,
    CanonicalMessage,
    CostEntry,
    Footage,
    GroundTruth,
    Line,
    Observation,
    RleMask,
    Rule,
    RuleSpec,
    TimeSpan,
)

CARTON_RULE: dict[str, Any] = {
    "entity": {"key": "carton", "phrase": "cardboard box"},
    "parts": {"tape": "packing tape", "label": "shipping label", "item": "product"},
    "checkpoint": {"kind": "enters", "zone": "sealing"},
    "requirements": [
        {"predicate": {"kind": "count_at_least", "part": "item", "n": 1}, "timing": "ever_before"},
        {"predicate": {"kind": "has", "part": "label"}, "timing": "at_checkpoint"},
        {"predicate": {"kind": "in_zone", "zone": "scale"}, "timing": "ever_before"},
    ],
    "order": [{"first": "tape", "then": "label"}],
    "dwell": {"zone": "sealing", "max_s": 20},
    "unsupported": ["and the box must be clean"],
}

SAMPLES: list[tuple[type[BaseModel], dict[str, Any]]] = [
    (RuleSpec, CARTON_RULE),
    (
        Observation,
        {
            "footage_id": "f1",
            "t": 3.2,
            "concept": "cardboard box",
            "box": {"x0": 0.1, "y0": 0.2, "x1": 0.4, "y1": 0.6},
            "score": 0.91,
            "object_id": 0,
            "mask": {"height": 2, "width": 3, "counts": [1, 4, 1]},
        },
    ),
    (
        GroundTruth,
        {
            "scenario": "carton_packing",
            "seed": 7,
            "duration_s": 60,
            "fps": 25,
            "width": 1280,
            "height": 720,
            "camera": {"position": [0, -3, 2], "look_at": [0, 0, 0.8], "focal_mm": 35},
            "zones": [
                {
                    "key": "sealing",
                    "source": "drawn",
                    "shape": {
                        "kind": "polygon",
                        "points": [
                            {"x": 0.5, "y": 0.2},
                            {"x": 0.7, "y": 0.2},
                            {"x": 0.6, "y": 0.8},
                        ],
                    },
                },
                {"key": "exit", "source": "implicit", "shape": {"kind": "view_exit"}},
            ],
            "items": [
                {
                    "id": "c1",
                    "order_id": "1043",
                    "t_spawn": 1.5,
                    "t_cross": {"sealing": 9.0},
                    "props": {"flaps_closed": False, "items": 2},
                    "expected_violations": ["missing_tape"],
                }
            ],
            "events": [{"kind": "jam", "span": {"start": 30, "end": 42}}],
        },
    ),
    (
        CostEntry,
        {
            "id": "c1",
            "at": "2026-10-28T14:00:29Z",
            "workspace_id": "w1",
            "provider": "meta",
            "model": "sam-3.1",
            "unit": "video_frame",
            "quantity": 60,
            "usd": "0.012",
        },
    ),
    (
        CanonicalMessage,
        {
            "channel": "telegram",
            "message_id": "m1",
            "chat_id": "c1",
            "sender_id": "u1",
            "received_at": "2026-10-28T14:00:00+05:30",
            "attachments": [{"kind": "video", "ref": "file-abc", "mime_type": "video/mp4"}],
        },
    ),
    (
        Line,
        {
            "id": "l1",
            "workspace_id": "w1",
            "name": "Packing line 2",
            "zones": [{"key": "exit", "source": "implicit", "shape": {"kind": "view_exit"}}],
            "created_at": "2026-10-28T14:00:00Z",
        },
    ),
    (
        Footage,
        {
            "id": "f1",
            "line_id": "l1",
            "content_hash": "sha256:ab12",
            "original_key": "footage/f1/original.mp4",
            "duration_s": 600,
            "fps": 25,
            "width": 1920,
            "height": 1080,
            "proxy": {
                "blob_key": "footage/f1/proxy.mp4",
                "fps": 5,
                "time_map_key": "footage/f1/time_map.json",
            },
            "status": "ready",
        },
    ),
    (
        Rule,
        {
            "id": "r1",
            "line_id": "l1",
            "text": "Every carton needs a label when it enters sealing.",
            "normalized_text": "every carton needs a label when it enters sealing",
            "spec": CARTON_RULE,
            "active": True,
            "created_at": "2026-10-28T14:00:00Z",
        },
    ),
]


@pytest.mark.parametrize(("model", "payload"), SAMPLES, ids=[m.__name__ for m, _ in SAMPLES])
def test_contracts_round_trip_through_json(model: type[BaseModel], payload: dict[str, Any]) -> None:
    parsed = model.model_validate(payload)

    assert model.model_validate_json(parsed.model_dump_json()) == parsed


def _rule(**changes: Any) -> dict[str, Any]:
    return CARTON_RULE | changes


REJECTED: list[tuple[str, type[BaseModel], dict[str, Any]]] = [
    (
        "unknown predicate",
        RuleSpec,
        _rule(requirements=[{"predicate": {"kind": "smells"}, "timing": "at_checkpoint"}]),
    ),
    ("unknown field", RuleSpec, _rule(severity="high")),
    (
        "undeclared part",
        RuleSpec,
        _rule(
            requirements=[{"predicate": {"kind": "has", "part": "lid"}, "timing": "ever_before"}]
        ),
    ),
    ("entity is also a part", RuleSpec, _rule(parts={"carton": "box"})),
    ("key not snake case", RuleSpec, _rule(checkpoint={"kind": "enters", "zone": "Sealing Zone"})),
    ("box corners swapped", Box, {"x0": 0.5, "y0": 0.1, "x1": 0.2, "y1": 0.3}),
    ("box outside frame", Box, {"x0": 0.5, "y0": 0.1, "x1": 1.2, "y1": 0.3}),
    ("mask runs too short", RleMask, {"height": 2, "width": 3, "counts": [1, 2]}),
    ("span ends before start", TimeSpan, {"start": 5, "end": 4}),
    (
        "naive timestamp",
        CostEntry,
        dict(SAMPLES[3][1], at="2026-10-28T14:00:29"),
    ),
    ("negative cost", CostEntry, dict(SAMPLES[3][1], usd="-0.01")),
    ("empty message", CanonicalMessage, dict(SAMPLES[4][1], attachments=[])),
    ("unknown footage status", Footage, dict(SAMPLES[6][1], status="done")),
    ("rule with an invalid spec", Rule, dict(SAMPLES[7][1], spec=_rule(parts={}))),
]


@pytest.mark.parametrize(
    ("model", "payload"), [r[1:] for r in REJECTED], ids=[r[0] for r in REJECTED]
)
def test_contracts_reject_bad_input(model: type[BaseModel], payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)
