from typing import Literal

from pydantic import AwareDatetime, Field

from rewind.contracts.base import Contract, Id
from rewind.contracts.geometry import Zone
from rewind.contracts.rules import RuleSpec


class Line(Contract):
    id: Id
    workspace_id: Id
    name: str = Field(min_length=1)
    zones: tuple[Zone, ...] = ()
    created_at: AwareDatetime


class Proxy(Contract):
    blob_key: Id
    fps: float = Field(gt=0)
    # Blob holding the footage time of each proxy frame.
    time_map_key: Id


class Footage(Contract):
    id: Id
    line_id: Id
    content_hash: Id
    original_key: Id
    duration_s: float = Field(gt=0)
    fps: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    proxy: Proxy | None = None
    status: Literal["uploaded", "processing", "ready", "failed"]


class Rule(Contract):
    id: Id
    line_id: Id
    text: str = Field(min_length=1)
    # The compile-cache key: the same normalized text never compiles twice.
    normalized_text: str = Field(min_length=1)
    spec: RuleSpec
    active: bool = True
    created_at: AwareDatetime
