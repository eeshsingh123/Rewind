import asyncio
import uuid
from collections import defaultdict
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Protocol

from rewind.contracts import Charge, CostEntry

EPOCH = datetime(2000, 1, 1, tzinfo=UTC)
UNCONFIRMED_SUFFIX = "_unconfirmed"


# The part of the Repository port the ledger needs, so a small local store can stand in.
class CostRecords(Protocol):
    async def add_cost(self, entry: CostEntry) -> None: ...

    async def spent_usd(self, since: datetime, workspace_id: str | None = None) -> Decimal: ...


@dataclass(frozen=True, slots=True)
class BudgetLimits:
    workspace_daily_usd: Decimal
    # Total spend across all workspaces, ever, after which no paid call is made.
    kill_switch_usd: Decimal


class BudgetExceededError(Exception):
    pass


def _utc_now() -> datetime:
    return datetime.now(UTC)


class Ledger:
    def __init__(
        self,
        records: CostRecords,
        limits: BudgetLimits,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._records = records
        self._limits = limits
        self._clock = clock
        self._lock = asyncio.Lock()
        # Estimates of calls in flight count as spent, so parallel calls cannot overshoot a cap.
        self._pending: defaultdict[str, Decimal] = defaultdict(Decimal)

    # Refuses a call whose estimate would cross a cap, and holds the estimate while it runs.
    @asynccontextmanager
    async def reserve(self, workspace_id: str, estimate_usd: Decimal) -> AsyncGenerator[None]:
        async with self._lock:
            await self._check(workspace_id, estimate_usd)
            self._pending[workspace_id] += estimate_usd
        try:
            yield
        finally:
            self._pending[workspace_id] -= estimate_usd

    async def record(
        self, workspace_id: str, job_id: str | None, charges: Sequence[Charge]
    ) -> None:
        for charge in charges:
            entry = CostEntry(
                **charge.model_dump(),
                id=uuid.uuid4().hex,
                at=self._clock(),
                workspace_id=workspace_id,
                job_id=job_id,
            )
            await self._records.add_cost(entry)

    # A failed call may still have been billed, so its estimate is kept, marked unconfirmed, and
    # counts toward the caps until someone reconciles it with the provider's bill.
    async def record_failure(
        self, workspace_id: str, job_id: str | None, estimate: Sequence[Charge]
    ) -> None:
        unconfirmed = [
            charge.model_copy(update={"unit": f"{charge.unit}{UNCONFIRMED_SUFFIX}"})
            for charge in estimate
        ]
        await self.record(workspace_id, job_id, unconfirmed)

    async def _check(self, workspace_id: str, estimate_usd: Decimal) -> None:
        now = self._clock()
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        total = await self._records.spent_usd(EPOCH) + sum(self._pending.values(), Decimal(0))
        if total + estimate_usd > self._limits.kill_switch_usd:
            raise BudgetExceededError(
                f"kill switch: ${total:.4f} spent of ${self._limits.kill_switch_usd}, "
                f"this call needs ${estimate_usd:.4f}"
            )
        today = await self._records.spent_usd(start_of_day, workspace_id)
        today += self._pending[workspace_id]
        if today + estimate_usd > self._limits.workspace_daily_usd:
            raise BudgetExceededError(
                f"daily cap: ${today:.4f} spent today of ${self._limits.workspace_daily_usd}, "
                f"this call needs ${estimate_usd:.4f}"
            )
