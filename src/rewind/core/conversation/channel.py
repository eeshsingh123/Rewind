from collections.abc import Mapping
from typing import Protocol

from rewind.contracts import Button, Reply, ReplyMedia
from rewind.core.ports import Channel


class ChatChannel(Channel, Protocol):
    async def show_progress(self, chat_id: str, key: str, text: str) -> None:
        """Updates the progress message with this key in place, or sends it if it is new."""
        ...


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

    async def progress(self, channel: str, chat_id: str, key: str, text: str) -> None:
        await self._channels[channel].show_progress(chat_id, key, text)
