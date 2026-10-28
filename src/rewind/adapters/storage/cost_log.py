import asyncio
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from rewind.contracts import CostEntry


# The ledger on local disk, one JSON line per entry, until the Postgres repository exists.
class JsonlCostLog:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()

    async def add_cost(self, entry: CostEntry) -> None:
        async with self._lock:
            await asyncio.to_thread(self._append, entry.model_dump_json() + "\n")

    async def spent_usd(self, since: datetime, workspace_id: str | None = None) -> Decimal:
        entries = await asyncio.to_thread(self.entries)
        return sum(
            (
                entry.usd
                for entry in entries
                if entry.at >= since and workspace_id in (None, entry.workspace_id)
            ),
            Decimal(0),
        )

    def entries(self) -> tuple[CostEntry, ...]:
        if not self._path.is_file():
            return ()
        lines = self._path.read_text(encoding="utf-8").splitlines()
        return tuple(CostEntry.model_validate_json(line) for line in lines if line)

    def _append(self, line: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as file:
            file.write(line)
