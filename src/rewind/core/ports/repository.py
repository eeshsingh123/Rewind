from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from typing import Protocol

from rewind.contracts import CostEntry, Event, Footage, Insight, Line, Rule, Violation


class RepositoryError(Exception):
    pass


class Repository(Protocol):
    async def save_line(self, line: Line) -> None:
        """Upserts by id."""
        ...

    async def get_line(self, line_id: str) -> Line | None: ...

    async def lines(self, workspace_id: str) -> tuple[Line, ...]: ...

    async def save_footage(self, footage: Footage) -> None:
        """Upserts by id."""
        ...

    async def get_footage(self, footage_id: str) -> Footage | None: ...

    async def line_footage(self, line_id: str) -> tuple[Footage, ...]: ...

    async def save_rule(self, rule: Rule) -> None:
        """Upserts by id."""
        ...

    async def get_rule(self, rule_id: str) -> Rule | None: ...

    async def rules(self, line_id: str) -> tuple[Rule, ...]: ...

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
