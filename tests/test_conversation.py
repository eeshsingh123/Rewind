from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rewind.adapters.jobs.in_process import InProcessJobRunner
from rewind.adapters.storage.sql import SqlRepository
from rewind.contracts import CanonicalMessage
from rewind.core.conversation import (
    INSTRUCTIONS,
    TURN_JOB,
    Agent,
    ChatRef,
    Conversation,
    ConversationSettings,
    Inbox,
    Readings,
    Toolbox,
    TurnLimits,
    create_line,
)
from rewind.core.ports import ToolCall
from tests.fakes import FakeBlobStore, FakeChannel, ScriptedModel, completion

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 10, 28, 9, 40, tzinfo=UTC)


@dataclass
class Setup:
    repository: SqlRepository
    jobs: InProcessJobRunner
    channel: FakeChannel
    model: ScriptedModel
    conversation: Conversation
    inbox: Inbox


@pytest.fixture
async def setup(tmp_path: Path) -> AsyncIterator[Setup]:
    repository = SqlRepository.from_url(f"sqlite+aiosqlite:///{tmp_path / 'rewind.db'}")
    await repository.create_tables()
    jobs = InProcessJobRunner()
    channel = FakeChannel()
    model = ScriptedModel([])
    conversation = Conversation(
        repository=repository, store=repository, blobs=FakeBlobStore(), jobs=jobs,
        channels={"telegram": channel},
        agent=Agent(INSTRUCTIONS, Toolbox.of([create_line(repository)]), TurnLimits()),
        model_for=lambda workspace_id: model, readings=Readings(),
        settings=ConversationSettings(frozenset(), "https://tunnel.test"), clock=lambda: NOW,
    )  # fmt: skip
    jobs.register(TURN_JOB, conversation.handle_job)
    inbox = Inbox(repository, jobs)
    yield Setup(repository, jobs, channel, model, conversation, inbox)
    await repository.close()


def _message(text: str) -> CanonicalMessage:
    return CanonicalMessage(
        channel="telegram", message_id="m1", chat_id="42", sender_id="7", received_at=NOW,
        text=text,
    )  # fmt: skip


async def test_a_repeated_delivery_is_answered_once(setup: Setup) -> None:
    setup.model.replies = [completion("Hi, I'm Rewind.")]

    assert await setup.inbox.receive(_message("hi")) is True
    assert await setup.inbox.receive(_message("hi")) is False
    await setup.jobs.drain()

    assert [reply.text for reply in setup.channel.sent] == ["Hi, I'm Rewind."]
    assert len(setup.model.requests) == 1


async def test_the_agent_calls_a_tool_then_answers_from_its_result(setup: Setup) -> None:
    setup.model.replies = [
        completion("", ToolCall("c1", "create_line", '{"name": "Packing line 2"}')),
        completion("Created *Packing line 2*."),
    ]

    await setup.conversation.handle(_message("New line: Packing line 2"))

    workspace_id = await setup.repository.workspace_for_chat(ChatRef("telegram", "42"))
    assert workspace_id is not None
    assert [line.name for line in await setup.repository.lines(workspace_id)] == ["Packing line 2"]
    tool_result = setup.model.requests[1].messages[-1]
    assert tool_result.role == "tool"
    assert "created line 'Packing line 2'" in tool_result.text
    assert setup.channel.sent[-1].text == "Created *Packing line 2*."
    history = await setup.repository.history(ChatRef("telegram", "42"), 10)
    assert [entry.text for entry in history] == [
        "New line: Packing line 2",
        "Created *Packing line 2*.",
    ]
