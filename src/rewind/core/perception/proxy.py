import hashlib
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rewind.contracts import Proxy, TimeSpan
from rewind.core.ports import BlobStore, VideoTools


@dataclass(frozen=True, slots=True)
class ProxyClip:
    proxy: Proxy
    width: int
    height: int
    # Footage time of each frame in the clip: the provider bills every one of them.
    frame_times: tuple[float, ...]


class ProxyMaker:
    def __init__(self, blobs: BlobStore, video: VideoTools, max_height: int) -> None:
        self._blobs = blobs
        self._video = video
        self._max_height = max_height

    # The clip of these spans at this fps, made once per footage content and then reused.
    async def ensure(
        self, content_hash: str, source_key: str, spans: Sequence[TimeSpan], fps: float
    ) -> ProxyClip:
        stem = f"proxies/{content_hash}/{fps:g}fps-{_spans_digest(spans)}"
        proxy = Proxy(blob_key=f"{stem}.mp4", fps=fps, time_map_key=f"{stem}.times.json")
        if await self._blobs.exists(proxy.blob_key) and await self._blobs.exists(
            proxy.time_map_key
        ):
            frame_times = tuple(json.loads(await self._blobs.read(proxy.time_map_key)))
        else:
            frame_times = await self._make(source_key, proxy, spans)
        info = await self._video.probe(await self._blobs.local_path(proxy.blob_key))
        return ProxyClip(proxy=proxy, width=info.width, height=info.height, frame_times=frame_times)

    async def _make(
        self, source_key: str, proxy: Proxy, spans: Sequence[TimeSpan]
    ) -> tuple[float, ...]:
        source = await self._blobs.local_path(source_key)
        with tempfile.TemporaryDirectory() as workdir:
            destination = Path(workdir) / "proxy.mp4"
            frame_times = await self._video.make_proxy(
                source, destination, spans, proxy.fps, self._max_height
            )
            await self._blobs.write_file(proxy.blob_key, destination)
        # Written last: its presence marks the proxy as complete.
        await self._blobs.write(proxy.time_map_key, json.dumps(frame_times).encode())
        return frame_times


def _spans_digest(spans: Sequence[TimeSpan]) -> str:
    text = ";".join(f"{span.start:.3f}-{span.end:.3f}" for span in spans)
    return hashlib.sha256(text.encode()).hexdigest()[:12]
