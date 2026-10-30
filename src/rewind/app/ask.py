from dataclasses import dataclass

import httpx

from rewind.adapters.models.muse import PRICES, MuseLanguageModel
from rewind.adapters.models.muse_voice import MuseVoice
from rewind.adapters.video.contact_sheet import OpenCvContactSheet
from rewind.adapters.video.photo import PillowPhotos
from rewind.adapters.video.proof_clip import OpenCvProofClips
from rewind.app.perception import Perception
from rewind.app.settings import Settings
from rewind.core.ask import (
    AskAgent,
    AskLimits,
    Proofs,
    Runs,
    Sample,
    Samples,
    Segmenters,
    Sessions,
    Toolset,
    Watcher,
)
from rewind.core.budget import (
    BudgetedLanguageModel,
    BudgetedTranscriber,
    BudgetScope,
    TokenPrices,
)
from rewind.core.perception import PartLooker
from rewind.core.ports import LanguageModel, Transcriber

_CONVEYOR = "stock/pexels_5903898_packages_conveyor_720p.mp4"


# Upload, ask, watch it work: what every channel calls.
@dataclass(frozen=True, slots=True)
class Ask:
    sessions: Sessions
    agent: AskAgent
    samples: Samples
    perception: Perception
    muse: MuseLanguageModel
    voice: MuseVoice
    prices: TokenPrices

    # Every paid call goes through the budget guard; callers never see the raw adapters.
    def model(self, scope: BudgetScope) -> LanguageModel:
        return BudgetedLanguageModel(self.muse, self.perception.ledger, self.prices, scope)

    def segmenters(self, scope: BudgetScope) -> Segmenters:
        return Segmenters(video=self.perception.video(scope), image=self.perception.image(scope))

    def hands(self, workspace_id: str, session_id: str) -> tuple[LanguageModel, Segmenters]:
        scope = BudgetScope(workspace_id, session_id)
        return self.model(scope), self.segmenters(scope)

    def transcriber(self, workspace_id: str) -> Transcriber:
        return BudgetedTranscriber(self.voice, self.perception.ledger, BudgetScope(workspace_id))

    def runs(self) -> Runs:
        return Runs(self.sessions, self.agent, self.hands)


def sample_footage(settings: Settings) -> tuple[Sample, ...]:
    data = settings.data_root
    return (
        Sample("jars", "Jars on a line", data / "sim/s1_jar_line-seed2/video.mp4"),
        Sample("cartons", "Packing cartons", data / "sim/s2_carton_packing-seed2/video.mp4"),
        Sample("conveyor", "Parcels on a conveyor", data / _CONVEYOR),
        Sample("shelf", "A shop shelf", data / "samples/shelf.jpg"),
    )


def build_ask(settings: Settings, client: httpx.AsyncClient, perception: Perception) -> Ask:
    if settings.meta_api_key is None:
        raise RuntimeError("META_API_KEY is not set")
    blobs = perception.blobs
    muse = MuseLanguageModel(
        client,
        settings.meta_api_key.get_secret_value(),
        settings.muse_model,
        blobs,
        perception.files,
    )
    price = PRICES[settings.muse_model]
    watcher = Watcher(blobs, perception.proxies)
    photos = PillowPhotos()
    sessions = Sessions(blobs, perception.ingest, watcher, photos, perception.video_tools)
    stills = PartLooker(blobs, perception.observations, perception.video_tools)
    tools = Toolset(
        blobs, watcher, perception.runner, stills, OpenCvContactSheet(), settings.track_fps
    ).tools()
    proofs = Proofs(blobs, perception.video_tools, OpenCvProofClips())
    return Ask(
        sessions=sessions,
        agent=AskAgent(sessions, tools, proofs, AskLimits()),
        samples=Samples(sample_footage(settings), blobs, perception.video_tools, photos),
        perception=perception,
        muse=muse,
        voice=MuseVoice(
            client, settings.meta_api_key.get_secret_value(), settings.meta_api_base_url
        ),
        prices=TokenPrices("meta", price.input, price.output),
    )
