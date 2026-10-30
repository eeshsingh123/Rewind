from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel
from sqlalchemy import Column, Index, MetaData, String, Table, Text, and_, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from rewind.contracts import CostEntry
from rewind.core.ports import ChatRef, RepositoryError

metadata = MetaData()


# Each record is stored whole as JSON, with only the columns it is looked up by.
def _records(name: str, *lookups: str) -> Table:
    table = Table(
        name,
        metadata,
        Column("id", String, primary_key=True),
        *(Column(lookup, String, nullable=False) for lookup in lookups),
        Column("body", Text, nullable=False),
    )
    for lookup in lookups:
        Index(f"ix_{name}_{lookup}", table.c[lookup])
    return table


costs_table = _records("costs", "workspace_id", "at")
claims_table = Table(
    "message_claims",
    metadata,
    Column("channel", String, primary_key=True),
    Column("message_id", String, primary_key=True),
)
chats_table = Table(
    "chats",
    metadata,
    Column("channel", String, primary_key=True),
    Column("chat_id", String, primary_key=True),
    Column("workspace_id", String, nullable=False, index=True),
)
chat_sessions_table = Table(
    "chat_sessions",
    metadata,
    Column("channel", String, primary_key=True),
    Column("chat_id", String, primary_key=True),
    Column("session_id", String, nullable=False),
)


# Fixed width, so text order is time order in both SQLite and Postgres.
def _instant(at: datetime) -> str:
    return at.astimezone(UTC).isoformat(timespec="microseconds")


class SqlRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    @classmethod
    def from_url(cls, url: str) -> "SqlRepository":
        # A paid result is lost if its cost cannot be written, so a write waits out another process
        # using the same SQLite file instead of giving up after SQLite's default 5 s.
        waits = {"timeout": 30} if url.startswith("sqlite") else {}
        return cls(create_async_engine(url, connect_args=waits))

    async def create_tables(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(metadata.create_all)

    async def close(self) -> None:
        await self._engine.dispose()

    async def claim_message(self, channel: str, message_id: str) -> bool:
        statement = (
            self._insert(claims_table)
            .values(channel=channel, message_id=message_id)
            .on_conflict_do_nothing()
        )
        async with self._begin() as connection:
            result = await connection.execute(statement)
        return result.rowcount == 1

    async def add_cost(self, entry: CostEntry) -> None:
        await self._upsert(
            costs_table, [entry], workspace_id=lambda r: r.workspace_id, at=lambda r: _instant(r.at)
        )

    async def spent_usd(self, since: datetime, workspace_id: str | None = None) -> Decimal:
        condition = costs_table.c.at >= _instant(since)
        if workspace_id is not None:
            condition = and_(condition, costs_table.c.workspace_id == workspace_id)
        entries = await self._find(costs_table, CostEntry, condition)
        return sum((entry.usd for entry in entries), Decimal(0))

    async def workspace_for_chat(self, chat: ChatRef) -> str | None:
        statement = select(chats_table.c.workspace_id).where(
            chats_table.c.channel == chat.channel, chats_table.c.chat_id == chat.chat_id
        )
        async with self._begin() as connection:
            return (await connection.execute(statement)).scalar_one_or_none()

    async def bind_chat(self, chat: ChatRef, workspace_id: str) -> None:
        statement = (
            self._insert(chats_table)
            .values(channel=chat.channel, chat_id=chat.chat_id, workspace_id=workspace_id)
            .on_conflict_do_update(
                index_elements=["channel", "chat_id"], set_={"workspace_id": workspace_id}
            )
        )
        async with self._begin() as connection:
            await connection.execute(statement)

    async def latest_session(self, chat: ChatRef) -> str | None:
        statement = select(chat_sessions_table.c.session_id).where(
            chat_sessions_table.c.channel == chat.channel,
            chat_sessions_table.c.chat_id == chat.chat_id,
        )
        async with self._begin() as connection:
            return (await connection.execute(statement)).scalar_one_or_none()

    async def set_latest_session(self, chat: ChatRef, session_id: str) -> None:
        statement = (
            self._insert(chat_sessions_table)
            .values(channel=chat.channel, chat_id=chat.chat_id, session_id=session_id)
            .on_conflict_do_update(
                index_elements=["channel", "chat_id"], set_={"session_id": session_id}
            )
        )
        async with self._begin() as connection:
            await connection.execute(statement)

    # Turns driver errors into the port's one error type.
    @asynccontextmanager
    async def _begin(self) -> AsyncIterator[AsyncConnection]:
        try:
            async with self._engine.begin() as connection:
                yield connection
        except SQLAlchemyError as error:
            raise RepositoryError(str(error)) from error

    def _insert(self, table: Table) -> Any:
        if self._engine.dialect.name == "postgresql":
            return postgres_insert(table)
        return sqlite_insert(table)

    async def _upsert(self, table: Table, records: Sequence[Any], **lookups: Any) -> None:
        if not records:
            return
        rows = [
            {
                "id": record.id,
                "body": record.model_dump_json(),
                **{name: read(record) for name, read in lookups.items()},
            }
            for record in records
        ]
        statement = self._insert(table).values(rows)
        statement = statement.on_conflict_do_update(
            index_elements=["id"],
            set_={name: statement.excluded[name] for name in ("body", *lookups)},
        )
        async with self._begin() as connection:
            await connection.execute(statement)

    async def _find[T: BaseModel](
        self, table: Table, model: type[T], condition: Any
    ) -> tuple[T, ...]:
        async with self._begin() as connection:
            bodies = (await connection.execute(select(table.c.body).where(condition))).scalars()
            return tuple(model.model_validate_json(body) for body in bodies)
