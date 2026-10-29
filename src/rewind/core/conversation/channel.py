import asyncio
import logging
from collections.abc import Mapping
from typing import Literal, Protocol

from rewind.contracts import Button, Reply, ReplyMedia
from rewind.core.ports import Channel, ChannelError, JobHandler, JobPayload

logger = logging.getLogger(__name__)

type ActivityKind = Literal["typing", "upload_video"]
# Telegram drops the indicator after about 5 s, so it is sent again a little sooner.
ACTIVITY_EVERY_S = 4.0
# A turn slower than this gets a message saying it is on it, not only "typing…".
ACK_AFTER_S = 2.0
ONE_MOMENT = "One moment…"


class ChatChannel(Channel, Protocol):
    async def show_progress(
        self, chat_id: str, key: str, text: str, buttons: tuple[Button, ...] = ()
    ) -> None:
        """Updates the progress message with this key in place, or sends it if it is new."""
        ...

    async def show_activity(self, chat_id: str, kind: ActivityKind) -> None:
        """Shows "typing…" or similar for a few seconds, or until the next message is sent."""
        ...


# Keeps the chat showing that work is going on until stopped. Only a cue: a failure is logged
# and never stops the work it decorates.
class Activity:
    def __init__(
        self, channel: ChatChannel, chat_id: str, every_s: float = ACTIVITY_EVERY_S
    ) -> None:
        self._channel = channel
        self._chat_id = chat_id
        self._every_s = every_s
        self._kind: ActivityKind = "typing"
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "Activity":
        self._task = asyncio.create_task(self._repeat())
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()

    async def show(self, kind: ActivityKind) -> None:
        self._kind = kind
        await self._show()

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        await asyncio.wait([self._task])
        self._task = None

    async def _repeat(self) -> None:
        while True:
            await self._show()
            await asyncio.sleep(self._every_s)

    async def _show(self) -> None:
        try:
            await self._channel.show_activity(self._chat_id, self._kind)
        except ChannelError:
            logger.warning("could not show %s in chat %s", self._kind, self._chat_id, exc_info=True)


# A turn that has not answered within a couple of seconds says so in a message of its own, and
# says what it is doing once a slow tool starts. The answer then takes that message's place.
class Ack:
    def __init__(self, channel: ChatChannel, chat_id: str, key: str, after_s: float) -> None:
        self._channel = channel
        self._chat_id = chat_id
        self.key = key
        self._after_s = after_s
        self._said = ""
        self._timer: asyncio.Task[None] | None = None

    @property
    def shown(self) -> bool:
        return bool(self._said)

    def start(self) -> None:
        self._timer = asyncio.create_task(self._later())

    async def status(self, text: str) -> None:
        await self.stop()
        await self._say(text)

    async def stop(self) -> None:
        if self._timer is None:
            return
        self._timer.cancel()
        await asyncio.wait([self._timer])
        self._timer = None

    async def _later(self) -> None:
        await asyncio.sleep(self._after_s)
        await self._say(ONE_MOMENT)

    async def _say(self, text: str) -> None:
        if text == self._said:
            return
        try:
            await self._channel.show_progress(self._chat_id, self.key, text)
        except ChannelError:
            logger.warning("could not acknowledge in chat %s", self._chat_id, exc_info=True)
            return
        self._said = text


# Sends to any chat by channel name; background jobs report through it.
class Outbox:
    def __init__(self, channels: Mapping[str, ChatChannel]) -> None:
        self._channels = channels

    async def notify(
        self,
        channel: str,
        chat_id: str,
        text: str,
        media: tuple[ReplyMedia, ...] = (),
        buttons: tuple[Button, ...] = (),
    ) -> None:
        reply = Reply(channel=channel, chat_id=chat_id, text=text, media=media, buttons=buttons)
        await self._channels[channel].send(reply)

    async def progress(
        self, channel: str, chat_id: str, key: str, text: str, buttons: tuple[Button, ...] = ()
    ) -> None:
        await self._channels[channel].show_progress(chat_id, key, text, buttons)

    def activity(self, channel: str, chat_id: str) -> Activity:
        return Activity(self._channels[channel], chat_id)


# A check can run for minutes; the chat shows "typing…" all along so it never looks idle.
def with_activity(outbox: Outbox, handler: JobHandler) -> JobHandler:
    async def run(payload: JobPayload) -> None:
        async with outbox.activity(str(payload["channel"]), str(payload["chat_id"])):
            await handler(payload)

    return run
