from dataclasses import dataclass

import httpx

from rewind.adapters.channels.telegram import TelegramChannel
from rewind.adapters.channels.whatsapp import WhatsAppChannel
from rewind.adapters.jobs.in_process import InProcessJobRunner
from rewind.adapters.storage.sql import SqlRepository
from rewind.app.ask import Ask
from rewind.app.settings import Settings
from rewind.core.ask import TURN_JOB, Chats, ChatSettings, Inbox, Runs, Voice
from rewind.core.ports import ChatChannel


@dataclass(frozen=True, slots=True)
class Chat:
    chats: Chats
    inbox: Inbox
    jobs: InProcessJobRunner
    telegram: TelegramChannel | None
    whatsapp: WhatsAppChannel | None


def build_chat(
    settings: Settings,
    client: httpx.AsyncClient,
    repository: SqlRepository,
    ask: Ask,
    runs: Runs,
) -> Chat:
    blobs = ask.perception.blobs
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
    chats = Chats(
        runs=runs,
        sessions=ask.sessions,
        store=repository,
        blobs=blobs,
        voice=Voice(blobs, ask.perception.video_tools, ask.transcriber),
        channels=channels,
        settings=ChatSettings(
            allowed_senders=frozenset(
                {f"telegram:{user}" for user in settings.telegram_allowed_user_ids}
                | {f"whatsapp:{number}" for number in settings.whatsapp_allowed_numbers}
            ),
            public_base_url=settings.public_base_url,
        ),
    )
    jobs = InProcessJobRunner()
    jobs.register(TURN_JOB, chats.handle_job)
    return Chat(chats, Inbox(repository, jobs), jobs, telegram, whatsapp)
