from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from rewind.adapters.storage.sql import SqlRepository
from rewind.contracts import CostEntry, Line, TimeSpan, Violation
from rewind.core.conversation import ChatRef, HistoryEntry

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 10, 28, 9, 0, tzinfo=UTC)
CHAT = ChatRef(channel="telegram", chat_id="42")


@pytest.fixture
async def repository(tmp_path: Path) -> AsyncIterator[SqlRepository]:
    repository = SqlRepository.from_url(f"sqlite+aiosqlite:///{tmp_path / 'rewind.db'}")
    await repository.create_tables()
    yield repository
    await repository.close()


async def test_a_repeated_delivery_is_claimed_once(repository: SqlRepository) -> None:
    assert await repository.claim_message("whatsapp", "wamid.1") is True
    assert await repository.claim_message("whatsapp", "wamid.1") is False
    assert await repository.claim_message("telegram", "wamid.1") is True


async def test_records_upsert_by_id(repository: SqlRepository) -> None:
    line = Line(id="l1", workspace_id="w1", name="Packing line 2", created_at=NOW)
    await repository.save_line(line)
    await repository.save_line(line.model_copy(update={"name": "Packing line 3"}))

    assert await repository.lines("w1") == (line.model_copy(update={"name": "Packing line 3"}),)
    assert await repository.get_line("missing") is None

    violation = Violation(
        id="v1", rule_id="r1", footage_id="f1", track_id="t1", t=3.0,
        reason="no label", clip=TimeSpan(start=0, end=6),
    )  # fmt: skip
    await repository.save_violations([violation, violation])
    assert await repository.violations("f1") == (violation,)


async def test_spend_is_summed_per_workspace_since_a_time(repository: SqlRepository) -> None:
    def entry(entry_id: str, workspace_id: str, at: datetime, usd: str) -> CostEntry:
        return CostEntry(
            id=entry_id, at=at, workspace_id=workspace_id, provider="meta", model="sam-3.1",
            unit="video_frame", quantity=100, usd=Decimal(usd),
        )  # fmt: skip

    await repository.add_cost(entry("c1", "w1", NOW - timedelta(days=1), "1.00"))
    await repository.add_cost(entry("c2", "w1", NOW, "0.25"))
    await repository.add_cost(entry("c3", "w2", NOW, "0.50"))

    assert await repository.spent_usd(NOW, "w1") == Decimal("0.25")
    assert await repository.spent_usd(NOW - timedelta(days=2)) == Decimal("1.75")


async def test_a_token_is_single_use_and_expires(repository: SqlRepository) -> None:
    await repository.put_token(
        "link", "4F7K29", {"workspace_id": "w1"}, NOW + timedelta(minutes=10)
    )
    await repository.put_token("link", "OLD", {"workspace_id": "w1"}, NOW - timedelta(minutes=1))

    assert await repository.peek_token("link", "4F7K29", NOW) == {"workspace_id": "w1"}
    assert await repository.take_token("link", "4F7K29", NOW) == {"workspace_id": "w1"}
    assert await repository.take_token("link", "4F7K29", NOW) is None
    assert await repository.take_token("link", "OLD", NOW) is None
    assert await repository.take_token("upload", "4F7K29", NOW) is None


async def test_chats_bind_to_workspaces_and_keep_recent_history(
    repository: SqlRepository,
) -> None:
    await repository.bind_chat(CHAT, "w1")
    await repository.bind_chat(CHAT, "w2")
    assert await repository.workspace_for_chat(CHAT) == "w2"
    assert await repository.chats("w2") == (CHAT,)

    for index in range(5):
        await repository.append_history(CHAT, HistoryEntry("user", f"m{index}"), NOW)
    assert [entry.text for entry in await repository.history(CHAT, 3)] == ["m2", "m3", "m4"]
