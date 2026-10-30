import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import APIRouter, FastAPI

from rewind.adapters.channels.whatsapp import (
    WEBHOOK_PATH,
    register_webhook,
    subscribe_account,
    webhook_router,
)
from rewind.adapters.storage.sql import SqlRepository
from rewind.app.ask import build_ask
from rewind.app.chat import Chat, build_chat
from rewind.app.perception import build_perception
from rewind.app.settings import Settings
from rewind.app.web import mount_page, web_router
from rewind.core.ask import MENU
from rewind.core.ports import ChannelError

logger = logging.getLogger(__name__)
router = APIRouter()
_TUNNEL_WAIT_S = 60


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # httpx logs request URLs at INFO, and the Telegram Bot API puts the token in the URL.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if settings.meta_api_key is None:
        logger.warning("META_API_KEY is not set: serving /health only, no chat channels")
        app = FastAPI(title="Rewind")
        app.include_router(router)
        return app

    client = httpx.AsyncClient(timeout=httpx.Timeout(120, connect=30))
    settings.data_root.mkdir(parents=True, exist_ok=True)
    repository = SqlRepository.from_url(settings.database_url_or_default)
    perception = build_perception(settings, client, repository)
    ask = build_ask(settings, client, perception)
    runs = ask.runs()
    chat = build_chat(settings, client, repository, ask, runs)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await repository.create_tables()
        tasks = _start_channels(settings, client, chat)
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await runs.close()
            await client.aclose()
            await repository.close()

    app = FastAPI(title="Rewind", lifespan=lifespan)
    app.state.settings = settings
    app.include_router(router)
    app.include_router(web_router(runs, ask.sessions, ask.samples, perception.blobs))
    mount_page(app)
    if chat.whatsapp is not None:
        if settings.whatsapp_app_secret is None or settings.whatsapp_verify_token is None:
            raise RuntimeError("WhatsApp needs REWIND_WHATSAPP_APP_SECRET and _VERIFY_TOKEN")
        app.include_router(
            webhook_router(
                settings.whatsapp_verify_token.get_secret_value(),
                settings.whatsapp_app_secret.get_secret_value(),
                chat.inbox.receive,
            )
        )
    return app


def _start_channels(
    settings: Settings, client: httpx.AsyncClient, chat: Chat
) -> list[asyncio.Task[None]]:
    tasks: list[asyncio.Task[None]] = []
    if chat.telegram is not None:
        if not settings.telegram_allowed_user_ids:
            logger.warning("Telegram has no allowlist: anyone who finds the bot can spend")
        tasks.append(asyncio.create_task(chat.telegram.poll(chat.inbox.receive)))
        tasks.append(asyncio.create_task(_set_telegram_commands(chat)))
        logger.info("Telegram polling started")
    app_secret, verify_token = settings.whatsapp_app_secret, settings.whatsapp_verify_token
    if chat.whatsapp is not None and settings.whatsapp_app_id and app_secret and verify_token:
        registration = _register_whatsapp(
            client, settings, settings.whatsapp_app_id,
            app_secret.get_secret_value(), verify_token.get_secret_value(),
        )  # fmt: skip
        tasks.append(asyncio.create_task(registration))
    return tasks


# The "/" menu is only a convenience: the bot works without it.
async def _set_telegram_commands(chat: Chat) -> None:
    if chat.telegram is None:
        return
    try:
        await chat.telegram.set_commands(MENU)
    except ChannelError:
        logger.exception("could not set the Telegram command menu")


# Meta checks the callback with a GET while registering, so this waits until the tunnel
# reaches this server.
async def _register_whatsapp(
    client: httpx.AsyncClient, settings: Settings, app_id: str, app_secret: str, verify_token: str
) -> None:
    base = settings.public_base_url.rstrip("/")
    for _ in range(_TUNNEL_WAIT_S):
        try:
            if (await client.get(f"{base}/health", timeout=5)).is_success:
                break
        except httpx.HTTPError:
            pass
        await asyncio.sleep(1)
    token, account_id = settings.whatsapp_token, settings.whatsapp_business_account_id
    try:
        await register_webhook(client, app_id, app_secret, f"{base}{WEBHOOK_PATH}", verify_token)
        if token is not None and account_id:
            await subscribe_account(client, token.get_secret_value(), account_id)
    except ChannelError:
        logger.exception("WhatsApp webhook registration failed")
        return
    logger.info("WhatsApp webhook registered at %s%s", base, WEBHOOK_PATH)
