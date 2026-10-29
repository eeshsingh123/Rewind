from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from rewind.adapters.channels.telegram import TelegramChannel
from rewind.adapters.channels.whatsapp import WhatsAppChannel
from rewind.adapters.jobs.in_process import InProcessJobRunner
from rewind.adapters.models.muse import PRICES, MuseLanguageModel
from rewind.adapters.storage.sql import SqlRepository
from rewind.adapters.video.proof_clip import OpenCvProofClips
from rewind.app.perception import Perception
from rewind.app.settings import Settings
from rewind.core.analysis import ANALYZE_JOB, INGEST_JOB, Analysis, AnalysisSettings
from rewind.core.budget import BudgetedLanguageModel, BudgetScope, TokenPrices
from rewind.core.conversation import (
    INSTRUCTIONS,
    TURN_JOB,
    Agent,
    Analyze,
    AnalyzeSettings,
    ChatChannel,
    Conversation,
    ConversationSettings,
    Findings,
    Inbox,
    Outbox,
    Readings,
    Toolbox,
    TurnLimits,
    add_rule,
    create_line,
    list_rules,
    rule_commands,
    upload_link,
    with_activity,
)
from rewind.core.ports import LanguageModel
from rewind.core.rules import RuleCompiler


@dataclass(frozen=True, slots=True)
class Chat:
    conversation: Conversation
    inbox: Inbox
    jobs: InProcessJobRunner
    telegram: TelegramChannel | None
    whatsapp: WhatsAppChannel | None


def _now() -> datetime:
    return datetime.now(UTC)


def build_chat(
    settings: Settings,
    client: httpx.AsyncClient,
    repository: SqlRepository,
    perception: Perception,
) -> Chat:
    if settings.meta_api_key is None:
        raise RuntimeError("META_API_KEY is not set")
    blobs = perception.blobs
    muse = MuseLanguageModel(
        client, settings.meta_api_key.get_secret_value(), settings.muse_model, blobs
    )
    # One price table for both the estimate and the charge: the Muse adapter's.
    muse_prices = PRICES[settings.muse_model]
    prices = TokenPrices("meta", muse_prices.input, muse_prices.output)

    def model_for(workspace_id: str) -> LanguageModel:
        return BudgetedLanguageModel(muse, perception.ledger, prices, BudgetScope(workspace_id))

    def compiler_for(workspace_id: str) -> RuleCompiler:
        return RuleCompiler(model_for(workspace_id), blobs)

    channels: dict[str, ChatChannel] = {}
    telegram = whatsapp = None
    if settings.telegram_bot_token is not None:
        telegram = TelegramChannel(client, settings.telegram_bot_token.get_secret_value(), blobs)
        channels[telegram.name] = telegram
    if settings.whatsapp_token is not None and settings.whatsapp_phone_number_id:
        whatsapp = WhatsAppChannel(
            client, settings.whatsapp_token.get_secret_value(),
            settings.whatsapp_phone_number_id, blobs,
        )  # fmt: skip
        channels[whatsapp.name] = whatsapp
    capabilities = {name: channel.capabilities for name, channel in channels.items()}

    # D-12: 8 GB of RAM, so heavy jobs run one at a time.
    jobs = InProcessJobRunner({INGEST_JOB: 1, ANALYZE_JOB: 1})
    outbox = Outbox(channels)
    analysis = Analysis(
        repository=repository,
        blobs=blobs,
        ingest=perception.ingest,
        planner=perception.planner,
        runner=perception.runner,
        observations=perception.observations,
        segmenters=perception,
        compiler_for=compiler_for,
        video=perception.video_tools,
        stills=perception.video_tools,
        clips=OpenCvProofClips(),
        notifier=outbox,
        settings=AnalysisSettings(fps=settings.track_fps),
    )
    readings = Readings()
    analyze = Analyze(
        repository, repository, analysis, readings, outbox, jobs,
        AnalyzeSettings(settings.confirm_above_usd, settings.workspace_daily_cap_usd), _now,
    )  # fmt: skip
    findings = Findings(repository, analysis, capabilities)
    conversation_settings = ConversationSettings(
        allowed_senders=frozenset(
            {f"telegram:{user}" for user in settings.telegram_allowed_user_ids}
            | {f"whatsapp:{number}" for number in settings.whatsapp_allowed_numbers}
        ),
        public_base_url=settings.public_base_url,
    )
    toolbox = Toolbox.of(
        [
            create_line(repository),
            list_rules(repository),
            add_rule(repository, compiler_for),
            upload_link(
                repository, repository, settings.public_base_url, conversation_settings.upload_ttl
            ),
            analyze.tool(),
            findings.violations_tool(),
            findings.clip_tool(),
        ]
    )
    conversation = Conversation(
        repository=repository,
        store=repository,
        blobs=blobs,
        jobs=jobs,
        channels=channels,
        agent=Agent(INSTRUCTIONS, toolbox, TurnLimits()),
        model_for=model_for,
        readings=readings,
        settings=conversation_settings,
        clock=_now,
        commands={
            "/confirm": analyze.confirm,
            "/cancel": analyze.cancel,
            "/clip": findings.clip_command,
            **rule_commands(repository),
        },
    )
    jobs.register(TURN_JOB, conversation.handle_job)
    jobs.register(INGEST_JOB, analyze.handle_ingest)
    jobs.register(ANALYZE_JOB, with_activity(outbox, analysis.handle_analyze))
    return Chat(conversation, Inbox(repository, jobs), jobs, telegram, whatsapp)
