# Live smoke run of the Telegram channel with the real agent and SAM; never run in CI.
#   uv run python scripts/smoke_telegram.py [--record-dir tests/recordings/telegram]
#       [--send /start --send "Anything wrong here?"] [--transcript smoke.md]
# Polls the bot like the app does, but hands every update to the inbox twice: the second must
# be dropped. Telegram's updates are recorded with ids and names replaced, for adapter tests.
# --send hands messages to the inbox as if the first allowed user had typed or tapped them, so
# a run needs nobody at the phone; --transcript writes down everything the bot sends or edits.
# The budget guard caps the spend; the ledger total is printed on exit (Ctrl+C).

import argparse
import asyncio
import json
import logging
import re
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from rewind.adapters.storage.sql import SqlRepository
from rewind.app.ask import build_ask
from rewind.app.chat import build_chat
from rewind.app.perception import build_perception
from rewind.app.settings import Settings
from rewind.contracts import CanonicalMessage
from rewind.core.ask import MENU
from rewind.core.budget import EPOCH

PERSON_KEYS = {"first_name": "Owner", "last_name": None, "username": "owner"}
FAKE_ID = 100200300
FILES = {"sendVideo", "sendPhoto"}
SENT = {"sendMessage", "editMessageText", "setMyCommands"} | FILES
_CAPTION = re.compile(rb'name="caption"\r\n\r\n(.*?)\r\n--', re.DOTALL)


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
    def __init__(self, directory: Path | None, transcript: Path | None) -> None:
        self._inner = httpx.AsyncHTTPTransport()
        self._directory = directory
        self._transcript = transcript
        self._started = time.monotonic()
        if transcript is not None:
            transcript.write_text("# What the bot sent\n", "utf-8")

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        method = request.url.path.rpartition("/")[2]
        if self._transcript is not None and method in SENT:
            # A video upload is streamed; it is read first so the caption can be written down.
            await request.aread()
        response = await self._inner.handle_async_request(request)
        body = await response.aread()
        if self._transcript is not None and method in SENT:
            self._write(method, request)
        if self._directory is not None and request.url.path.endswith("/getUpdates"):
            payload = json.loads(body)
            for update in payload.get("result", []):
                self._save(update)
        return httpx.Response(
            response.status_code, headers=response.headers, content=body, request=request
        )

    def _write(self, method: str, request: httpx.Request) -> None:
        assert self._transcript is not None
        at = f"{time.monotonic() - self._started:6.1f} s"
        if method in FILES:
            found = _CAPTION.search(request.content)
            text = found.group(1).decode() if found else ""
            buttons = "inline_keyboard" in request.content.decode(errors="ignore")
        else:
            payload = json.loads(request.content)
            text = payload.get("text") or json.dumps(payload.get("commands"), ensure_ascii=False)
            markup = payload.get("reply_markup") or {}
            buttons = [b["text"] for row in markup.get("inline_keyboard", []) for b in row]
        entry = f"\n## {at} · {method}\n\n{text}\n"
        if buttons:
            entry += f"\nButtons: {buttons}\n"
        with self._transcript.open("a", encoding="utf-8") as file:
            file.write(entry)

    def _save(self, update: dict[str, Any]) -> None:
        assert self._directory is not None
        self._directory.mkdir(parents=True, exist_ok=True)
        kind = next((k for k in ("callback_query", "message") if k in update), "other")
        message = update.get("message", {})
        files = ("video", "video_note", "photo", "voice", "audio", "document")
        kind += "".join(f"_{file}" for file in files if file in message)
        name = f"{update['update_id']}_{kind}.json"
        (self._directory / name).write_text(json.dumps(redact(update), indent=2), "utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/smoke-telegram"))
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--cap-usd", type=Decimal, default=Decimal("0.50"))
    parser.add_argument("--send", action="append", default=[])
    parser.add_argument("--transcript", type=Path)
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
    transport = RecordingTransport(args.record_dir, args.transcript)
    async with httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(120)) as client:
        ask = build_ask(settings, client, build_perception(settings, client, repository))
        chat = build_chat(settings, client, repository, ask, ask.runs())
        assert chat.telegram is not None

        async def receive_twice(message: CanonicalMessage) -> bool:
            first = await chat.inbox.receive(message)
            second = await chat.inbox.receive(message)
            print(f"update {message.message_id}: first={first} repeat={second}", flush=True)
            return first

        await chat.telegram.set_commands(MENU)
        print("polling; send the bot a message (Ctrl+C to stop)", flush=True)
        owner = sorted(settings.telegram_allowed_user_ids)[:1]
        for index, text in enumerate(args.send):
            if not owner:
                raise SystemExit("--send needs REWIND_TELEGRAM_ALLOWED_USER_IDS")
            message = CanonicalMessage(
                channel="telegram", message_id=f"smoke-{time.time_ns()}-{index}",
                chat_id=owner[0], sender_id=owner[0], received_at=datetime.now(UTC), text=text,
            )  # fmt: skip
            await receive_twice(message)
            # The next one waits for the bot to answer, as a person would.
            await chat.jobs.drain()
        try:
            await chat.telegram.poll(receive_twice)
        finally:
            await chat.jobs.drain()
            print(f"ledger: ${await repository.spent_usd(EPOCH)}", flush=True)
            await repository.close()


if __name__ == "__main__":
    asyncio.run(main())
