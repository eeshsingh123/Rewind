from pathlib import Path
from typing import Protocol

from rewind.contracts import Attachment, ChannelCapabilities, Reply


class ChannelError(Exception):
    pass


class Channel(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def capabilities(self) -> ChannelCapabilities: ...

    async def send(self, reply: Reply) -> None: ...

    async def fetch_attachment(self, attachment: Attachment, destination: Path) -> None: ...
