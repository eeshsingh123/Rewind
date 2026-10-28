# Live smoke run of the Telegram channel with the real agent, SAM and rules; never run in CI.
#   uv run python scripts/smoke_telegram.py [--record-dir tests/recordings/telegram]
# Polls the bot like the app does, but hands every update to the inbox twice: the second must
# be dropped. Telegram's updates are recorded with ids and names replaced, for adapter tests.
# The budget guard caps the spend; the ledger total is printed on exit (Ctrl+C).

import argparse
import asyncio
import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from rewind.adapters.storage.sql import SqlRepository
from rewind.app.chat import build_chat
from rewind.app.perception import build_perception
from rewind.app.settings import Settings
from rewind.contracts import CanonicalMessage
from rewind.core.budget import EPOCH

PERSON_KEYS = {"first_name": "Owner", "last_name": None, "username": "owner"}
FAKE_ID = 100200300


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in PERSON_KEYS:
                if PERSON_KEYS[key] is not None:
                    out[key] = PERSON_KEYS[key]
            elif key == "id" and isinstance(item, int):
                out[key] = FAKE_ID
            else:
                out[key] = redact(item)
        return out
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


# Saves getUpdates results that carry updates. Never the URL: it holds the bot token.
class RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self, directory: Path | None) -> None:
        self._inner = httpx.AsyncHTTPTransport()
        self._directory = directory

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        body = await response.aread()
        if self._directory is not None and request.url.path.endswith("/getUpdates"):
            payload = json.loads(body)
            for update in payload.get("result", []):
                self._save(update)
        return httpx.Response(
            response.status_code, headers=response.headers, content=body, request=request
        )

    def _save(self, update: dict[str, Any]) -> None:
        assert self._directory is not None
        self._directory.mkdir(parents=True, exist_ok=True)
        kind = next((k for k in ("callback_query", "message") if k in update), "other")
        message = update.get("message", {})
        kind += "_video" if "video" in message or "document" in message else ""
        name = f"{update['update_id']}_{kind}.json"
        (self._directory / name).write_text(json.dumps(redact(update), indent=2), "utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/smoke-telegram"))
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--cap-usd", type=Decimal, default=Decimal("0.50"))
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # httpx logs URLs at INFO, and the Bot API puts the token in the URL.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = Settings(
        data_root=args.data_root,
        kill_switch_usd=args.cap_usd,
        workspace_daily_cap_usd=args.cap_usd,
    )
    if settings.telegram_bot_token is None:
        raise SystemExit("REWIND_TELEGRAM_BOT_TOKEN is not set")
    args.data_root.mkdir(parents=True, exist_ok=True)
    repository = SqlRepository.from_url(settings.database_url_or_default)
    await repository.create_tables()
    transport = RecordingTransport(args.record_dir)
    async with httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(120)) as client:
        perception = build_perception(settings, client, repository)
        chat = build_chat(settings, client, repository, perception)
        assert chat.telegram is not None

        async def receive_twice(message: CanonicalMessage) -> bool:
            first = await chat.inbox.receive(message)
            second = await chat.inbox.receive(message)
            print(f"update {message.message_id}: first={first} repeat={second}", flush=True)
            return first

        print("polling; send the bot a message (Ctrl+C to stop)", flush=True)
        try:
            await chat.telegram.poll(receive_twice)
        finally:
            await chat.jobs.drain()
            print(f"ledger: ${await repository.spent_usd(EPOCH)}", flush=True)
            await repository.close()


if __name__ == "__main__":
    asyncio.run(main())
