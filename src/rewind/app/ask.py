from dataclasses import dataclass

import httpx

from rewind.adapters.models.muse import PRICES, MuseLanguageModel
from rewind.adapters.video.contact_sheet import OpenCvContactSheet
from rewind.adapters.video.proof_clip import OpenCvProofClips
from rewind.app.perception import Perception
from rewind.app.settings import Settings
from rewind.core.ask import AskAgent, AskLimits, ProofClips, Sessions, Toolset, Watcher
from rewind.core.budget import BudgetedLanguageModel, BudgetScope, TokenPrices
from rewind.core.ports import LanguageModel, Segmenter


# Upload, ask, watch it work: what every channel calls.
@dataclass(frozen=True, slots=True)
class Ask:
    sessions: Sessions
    agent: AskAgent
    perception: Perception
    muse: MuseLanguageModel
    prices: TokenPrices

    # Every paid call goes through the budget guard; callers never see the raw adapters.
    def model(self, scope: BudgetScope) -> LanguageModel:
        return BudgetedLanguageModel(self.muse, self.perception.ledger, self.prices, scope)

    def segmenter(self, scope: BudgetScope) -> Segmenter:
        return self.perception.video(scope)


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
    sessions = Sessions(blobs, perception.ingest, watcher)
    tools = Toolset(
        blobs, watcher, perception.runner, OpenCvContactSheet(), settings.track_fps
    ).tools()
    clips = ProofClips(blobs, perception.video_tools, OpenCvProofClips())
    return Ask(
        sessions=sessions,
        agent=AskAgent(sessions, tools, clips, AskLimits()),
        perception=perception,
        muse=muse,
        prices=TokenPrices("meta", price.input, price.output),
    )
