import secrets
from dataclasses import dataclass

import httpx

from rewind.adapters.segmenters.meta_api import MetaApi
from rewind.adapters.segmenters.meta_image import MetaImageSegmenter
from rewind.adapters.segmenters.meta_video import MetaVideoSegmenter
from rewind.adapters.storage.cost_log import JsonlCostLog
from rewind.adapters.storage.local_disk import LocalDiskBlobStore
from rewind.adapters.storage.parquet_observations import ParquetObservationStore
from rewind.adapters.video.ffmpeg import FfmpegVideoTools
from rewind.adapters.video.opencv import OpenCvActivityProbe
from rewind.app.settings import Settings
from rewind.core.budget import BudgetedSegmenter, BudgetLimits, BudgetScope, Ledger
from rewind.core.perception import (
    STRATEGIES,
    Ingest,
    PerceptionRunner,
    Planner,
    ProxyMaker,
)
from rewind.core.ports import Segmenter


@dataclass(frozen=True, slots=True)
class Perception:
    blobs: LocalDiskBlobStore
    proxies: ProxyMaker
    ingest: Ingest
    planner: Planner
    runner: PerceptionRunner
    ledger: Ledger
    cost_log: JsonlCostLog
    video_segmenter: MetaVideoSegmenter
    image_segmenter: MetaImageSegmenter

    # Every paid call goes through the budget guard; callers never see the raw adapters.
    def video(self, scope: BudgetScope) -> Segmenter:
        return BudgetedSegmenter(self.video_segmenter, self.ledger, scope)

    def image(self, scope: BudgetScope) -> Segmenter:
        return BudgetedSegmenter(self.image_segmenter, self.ledger, scope)


def build_perception(settings: Settings, client: httpx.AsyncClient) -> Perception:
    if settings.meta_api_key is None:
        raise RuntimeError("META_API_KEY is not set")
    signing_key = (
        settings.media_signing_key.get_secret_value().encode()
        if settings.media_signing_key
        else secrets.token_bytes(32)
    )
    blobs = LocalDiskBlobStore(settings.data_root / "blobs", settings.public_base_url, signing_key)
    video_tools = FfmpegVideoTools()
    proxies = ProxyMaker(blobs, video_tools, settings.proxy_max_height)
    api = MetaApi(client, settings.meta_api_key.get_secret_value(), settings.meta_api_base_url)
    cost_log = JsonlCostLog(settings.data_root / "ledger.jsonl")
    limits = BudgetLimits(
        workspace_daily_usd=settings.workspace_daily_cap_usd,
        kill_switch_usd=settings.kill_switch_usd,
    )
    return Perception(
        blobs=blobs,
        proxies=proxies,
        ingest=Ingest(blobs, video_tools, OpenCvActivityProbe(), proxies, settings.track_fps),
        planner=Planner(STRATEGIES),
        runner=PerceptionRunner(proxies, ParquetObservationStore(blobs)),
        ledger=Ledger(cost_log, limits),
        cost_log=cost_log,
        video_segmenter=MetaVideoSegmenter(api, blobs, settings.meta_file_ttl_s),
        image_segmenter=MetaImageSegmenter(api, blobs),
    )
