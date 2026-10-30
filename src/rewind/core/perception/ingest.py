import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter

from rewind.contracts import Footage, TimeSpan
from rewind.core.perception.activity import ActivityProbe, IdleSkip, active_spans
from rewind.core.perception.proxy import ProxyMaker
from rewind.core.ports import BlobStore, VideoTools

_SPANS = TypeAdapter(tuple[TimeSpan, ...])


@dataclass(frozen=True, slots=True)
class Ingested:
    footage: Footage
    # Where something moves; only these spans are ever sent to a paid model.
    active_spans: tuple[TimeSpan, ...]


class Ingest:
    def __init__(
        self,
        blobs: BlobStore,
        video: VideoTools,
        activity: ActivityProbe,
        proxies: ProxyMaker,
        proxy_fps: float,
        idle_skip: IdleSkip | None = None,
    ) -> None:
        self._blobs = blobs
        self._video = video
        self._activity = activity
        self._proxies = proxies
        self._proxy_fps = proxy_fps
        self._idle_skip = idle_skip or IdleSkip()

    async def ingest(self, footage_id: str, line_id: str, original_key: str) -> Ingested:
        source = await self._blobs.local_path(original_key)
        content_hash = await asyncio.to_thread(_sha256, source)
        info = await self._video.probe(source)
        spans = await self._active_spans(source, content_hash, info.duration_s)
        proxy = None
        if spans:
            clip = await self._proxies.ensure(content_hash, original_key, spans, self._proxy_fps)
            proxy = clip.proxy
        footage = Footage(
            id=footage_id,
            line_id=line_id,
            content_hash=content_hash,
            original_key=original_key,
            duration_s=info.duration_s,
            fps=info.fps,
            width=info.width,
            height=info.height,
            proxy=proxy,
            status="ready",
        )
        return Ingested(footage=footage, active_spans=spans)

    async def _active_spans(
        self, source: Path, content_hash: str, duration_s: float
    ) -> tuple[TimeSpan, ...]:
        key = f"activity/{content_hash}.json"
        if await self._blobs.exists(key):
            return _SPANS.validate_json(await self._blobs.read(key))
        samples = await self._activity.sample(source, self._idle_skip.sample_fps)
        spans = active_spans(samples, duration_s, self._idle_skip)
        await self._blobs.write(key, json.dumps([span.model_dump() for span in spans]).encode())
        return spans


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
