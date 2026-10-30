import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, JsonValue
from sqlalchemy import (
    Column,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
    delete,
    select,
)
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from rewind.contracts import CostEntry, Event, Footage, Insight, Line, Rule, Violation
from rewind.core.conversation import HistoryEntry
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


lines_table = _records("lines", "workspace_id")
footage_table = _records("footage", "line_id")
rules_table = _records("rules", "line_id")
violations_table = _records("violations", "footage_id")
events_table = _records("events", "footage_id")
insights_table = _records("insights", "footage_id")
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
tokens_table = Table(
    "tokens",
    metadata,
    Column("kind", String, primary_key=True),
    Column("token", String, primary_key=True),
    Column("payload", Text, nullable=False),
    Column("expires_at", String, nullable=False),
)
history_table = Table(
    "history",
    metadata,
    Column("seq", Integer, primary_key=True, autoincrement=True),
    Column("channel", String, nullable=False),
    Column("chat_id", String, nullable=False),
    Column("role", String, nullable=False),
    Column("text", Text, nullable=False),
    Column("at", String, nullable=False),
    Index("ix_history_chat", "channel", "chat_id", "seq"),
)


# Fixed width, so text order is time order in both SQLite and Postgres.
def _instant(at: datetime) -> str:
    return at.astimezone(UTC).isoformat(timespec="microseconds")


class SqlRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    @classmethod
    def from_url(cls, url: str) -> "SqlRepository":
        return cls(create_async_engine(url))

    async def create_tables(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(metadata.create_all)

    async def close(self) -> None:
        await self._engine.dispose()

    async def save_line(self, line: Line) -> None:
        await self._upsert(lines_table, [line], workspace_id=lambda r: r.workspace_id)

    async def get_line(self, line_id: str) -> Line | None:
        return await self._get(lines_table, Line, line_id)

    async def lines(self, workspace_id: str) -> tuple[Line, ...]:
        return await self._find(lines_table, Line, lines_table.c.workspace_id == workspace_id)

    async def save_footage(self, footage: Footage) -> None:
        await self._upsert(footage_table, [footage], line_id=lambda r: r.line_id)

    async def get_footage(self, footage_id: str) -> Footage | None:
        return await self._get(footage_table, Footage, footage_id)

    async def line_footage(self, line_id: str) -> tuple[Footage, ...]:
        return await self._find(footage_table, Footage, footage_table.c.line_id == line_id)

    async def save_rule(self, rule: Rule) -> None:
        await self._upsert(rules_table, [rule], line_id=lambda r: r.line_id)

    async def get_rule(self, rule_id: str) -> Rule | None:
        return await self._get(rules_table, Rule, rule_id)

    async def rules(self, line_id: str) -> tuple[Rule, ...]:
        found = await self._find(rules_table, Rule, rules_table.c.line_id == line_id)
        return tuple(sorted(found, key=lambda rule: rule.created_at))

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

    async def save_violations(self, violations: Sequence[Violation]) -> None:
        await self._upsert(violations_table, violations, footage_id=lambda r: r.footage_id)

    async def replace_violations(
        self, footage_id: str, rule_ids: Sequence[str], violations: Sequence[Violation]
    ) -> None:
        old = await self.violations(footage_id)
        stale = [v.id for v in old if v.rule_id in rule_ids]
        if stale:
            async with self._begin() as connection:
                await connection.execute(
                    delete(violations_table).where(violations_table.c.id.in_(stale))
                )
        await self.save_violations(violations)

    async def violations(self, footage_id: str) -> tuple[Violation, ...]:
        found = await self._find(
            violations_table, Violation, violations_table.c.footage_id == footage_id
        )
        return tuple(sorted(found, key=lambda violation: (violation.t, violation.id)))

    async def save_events(self, events: Sequence[Event]) -> None:
        await self._upsert(events_table, events, footage_id=lambda r: r.footage_id)

    async def events(self, footage_id: str) -> tuple[Event, ...]:
        return await self._find(events_table, Event, events_table.c.footage_id == footage_id)

    async def save_insights(self, insights: Sequence[Insight]) -> None:
        await self._upsert(insights_table, insights, footage_id=lambda r: r.footage_id)

    async def insights(self, footage_id: str) -> tuple[Insight, ...]:
        return await self._find(insights_table, Insight, insights_table.c.footage_id == footage_id)

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

    async def chats(self, workspace_id: str) -> tuple[ChatRef, ...]:
        statement = select(chats_table.c.channel, chats_table.c.chat_id).where(
            chats_table.c.workspace_id == workspace_id
        )
        async with self._begin() as connection:
            rows = (await connection.execute(statement)).all()
        return tuple(ChatRef(channel=row.channel, chat_id=row.chat_id) for row in rows)

    async def put_token(
        self, kind: str, token: str, payload: dict[str, JsonValue], expires_at: datetime
    ) -> None:
        values = {"payload": json.dumps(payload), "expires_at": _instant(expires_at)}
        statement = (
            self._insert(tokens_table)
            .values(kind=kind, token=token, **values)
            .on_conflict_do_update(index_elements=["kind", "token"], set_=values)
        )
        async with self._begin() as connection:
            await connection.execute(statement)

    async def take_token(self, kind: str, token: str, now: datetime) -> dict[str, JsonValue] | None:
        # Deleting with RETURNING makes the token single-use even when two requests race.
        statement = (
            delete(tokens_table)
            .where(tokens_table.c.kind == kind, tokens_table.c.token == token)
            .returning(tokens_table.c.payload, tokens_table.c.expires_at)
        )
        async with self._begin() as connection:
            row = (await connection.execute(statement)).one_or_none()
        if row is None or row.expires_at <= _instant(now):
            return None
        return json.loads(row.payload)

    async def peek_token(self, kind: str, token: str, now: datetime) -> dict[str, JsonValue] | None:
        statement = select(tokens_table.c.payload).where(
            tokens_table.c.kind == kind,
            tokens_table.c.token == token,
            tokens_table.c.expires_at > _instant(now),
        )
        async with self._begin() as connection:
            payload = (await connection.execute(statement)).scalar_one_or_none()
        return None if payload is None else json.loads(payload)

    async def append_history(self, chat: ChatRef, entry: HistoryEntry, at: datetime) -> None:
        statement = history_table.insert().values(
            channel=chat.channel,
            chat_id=chat.chat_id,
            role=entry.role,
            text=entry.text,
            at=_instant(at),
        )
        async with self._begin() as connection:
            await connection.execute(statement)

    async def history(self, chat: ChatRef, limit: int) -> tuple[HistoryEntry, ...]:
        statement = (
            select(history_table.c.role, history_table.c.text)
            .where(history_table.c.channel == chat.channel, history_table.c.chat_id == chat.chat_id)
            .order_by(history_table.c.seq.desc())
            .limit(limit)
        )
        async with self._begin() as connection:
            rows = (await connection.execute(statement)).all()
        return tuple(HistoryEntry(role=row.role, text=row.text) for row in reversed(rows))

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

    async def _get[T: BaseModel](self, table: Table, model: type[T], record_id: str) -> T | None:
        async with self._begin() as connection:
            body = (
                await connection.execute(select(table.c.body).where(table.c.id == record_id))
            ).scalar_one_or_none()
        return None if body is None else model.model_validate_json(body)

    async def _find[T: BaseModel](
        self, table: Table, model: type[T], condition: Any
    ) -> tuple[T, ...]:
        async with self._begin() as connection:
            bodies = (await connection.execute(select(table.c.body).where(condition))).scalars()
            return tuple(model.model_validate_json(body) for body in bodies)
