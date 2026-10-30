from datetime import datetime
from decimal import Decimal
from typing import Protocol

from rewind.contracts import CostEntry


class RepositoryError(Exception):
    pass


class Repository(Protocol):
    async def claim_message(self, channel: str, message_id: str) -> bool:
        """True the first time a message is seen, False for a repeated delivery."""
        ...

    async def add_cost(self, entry: CostEntry) -> None: ...

    async def spent_usd(self, since: datetime, workspace_id: str | None = None) -> Decimal:
        """Spend since a time, for one workspace, or for all of them when None."""
        ...
