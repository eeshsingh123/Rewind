from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Protocol

from rewind.contracts import CostEntry, Event, Insight, Violation


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

    async def save_violations(self, violations: Sequence[Violation]) -> None:
        """Upserts by id, so re-running a rule never duplicates its findings."""
        ...

    async def violations(self, footage_id: str) -> tuple[Violation, ...]: ...

    async def save_events(self, events: Sequence[Event]) -> None:
        """Upserts by id."""
        ...

    async def events(self, footage_id: str) -> tuple[Event, ...]: ...

    async def save_insights(self, insights: Sequence[Insight]) -> None:
        """Upserts by id."""
        ...

    async def insights(self, footage_id: str) -> tuple[Insight, ...]: ...
