from typing import Literal

from pydantic import Field

from rewind.contracts.base import Contract, Id


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
