import asyncio
import logging
from collections.abc import Mapping

from rewind.contracts import Button, Reply, ReplyMedia
from rewind.core.ask.chat import Activity
from rewind.core.ports import ChannelError, ChatChannel, JobHandler, JobPayload

logger = logging.getLogger(__name__)

# A turn slower than this gets a message saying it is on it, not only "typing…".
ACK_AFTER_S = 2.0
ONE_MOMENT = "One moment…"


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
