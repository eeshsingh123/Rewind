from typing import Protocol

from rewind.core.ports import Channel


class ChatChannel(Channel, Protocol):
    async def show_progress(self, chat_id: str, key: str, text: str) -> None:
        """Updates the progress message with this key in place, or sends it if it is new."""
        ...
