import asyncio
import hashlib
import io
from collections import defaultdict
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from rewind.contracts import Box, Observation, RleMask
from rewind.core.perception.cache import CachedFrames, CacheKey
from rewind.core.ports import BlobStore

# One row per observation, plus one row with a null score for each frame where nothing was found.
SCHEMA = pa.schema(
    [
        ("frame_idx", pa.int64()),
        ("t", pa.float64()),
        ("concept", pa.string()),
        ("score", pa.float64()),
        ("x0", pa.float64()),
        ("y0", pa.float64()),
        ("x1", pa.float64()),
        ("y1", pa.float64()),
        ("object_id", pa.int64()),
        ("mask_height", pa.int32()),
        ("mask_width", pa.int32()),
        ("mask_counts", pa.list_(pa.int64())),
    ]
)


class ParquetObservationStore:
    def __init__(self, blobs: BlobStore) -> None:
        self._blobs = blobs

    async def load(self, key: CacheKey, footage_id: str) -> CachedFrames:
        blob_key = _blob_key(key)
        if not await self._blobs.exists(blob_key):
            return CachedFrames(frames={})
        data = await self._blobs.read(blob_key)
        return await asyncio.to_thread(_decode, data, footage_id)

    async def save(self, key: CacheKey, cached: CachedFrames) -> None:
        data = await asyncio.to_thread(_encode, key, cached)
        await self._blobs.write(_blob_key(key), data)


def _decode(data: bytes, footage_id: str) -> CachedFrames:
    table = pq.read_table(io.BytesIO(data), schema=SCHEMA)
    frames: defaultdict[int, list[Observation]] = defaultdict(list)
    for row in table.to_pylist():
        found = frames[row["frame_idx"]]
        if row["score"] is not None:
            found.append(_observation(row, footage_id))
    return CachedFrames(frames={index: tuple(found) for index, found in frames.items()})


def _encode(key: CacheKey, cached: CachedFrames) -> bytes:
    rows: list[dict[str, Any]] = []
    for index, observations in sorted(cached.frames.items()):
        if not observations:
            rows.append({"frame_idx": index, "concept": key.concept})
        rows.extend(_row(index, observation) for observation in observations)
    buffer = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), buffer)
    return buffer.getvalue()


def _blob_key(key: CacheKey) -> str:
    concept = hashlib.sha256(key.concept.encode()).hexdigest()[:16]
    return f"observations/{key.content_hash}/{key.model_version}/{concept}.parquet"


def _row(index: int, observation: Observation) -> dict[str, Any]:
    mask = observation.mask
    return {
        "frame_idx": index,
        "t": observation.t,
        "concept": observation.concept,
        "score": observation.score,
        "x0": observation.box.x0,
        "y0": observation.box.y0,
        "x1": observation.box.x1,
        "y1": observation.box.y1,
        "object_id": observation.object_id,
        "mask_height": mask.height if mask else None,
        "mask_width": mask.width if mask else None,
        "mask_counts": list(mask.counts) if mask else None,
    }


def _observation(row: dict[str, Any], footage_id: str) -> Observation:
    mask = None
    if row["mask_counts"] is not None:
        mask = RleMask(
            height=row["mask_height"], width=row["mask_width"], counts=tuple(row["mask_counts"])
        )
    return Observation(
        footage_id=footage_id,
        t=row["t"],
        concept=row["concept"],
        box=Box(x0=row["x0"], y0=row["y0"], x1=row["x1"], y1=row["y1"]),
        score=row["score"],
        object_id=row["object_id"],
        mask=mask,
    )
