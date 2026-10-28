from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from rewind.adapters.jobs.in_process import InProcessJobRunner
from rewind.adapters.storage.sql import SqlRepository
from rewind.contracts import Attachment, CanonicalMessage, ChannelCapabilities, Charge, Reply
from rewind.core.conversation import (
    INGEST_JOB,
    INSTRUCTIONS,
    TURN_JOB,
    Agent,
    AgentReply,
    AttachmentTooLargeError,
    ChatRef,
    Conversation,
    ConversationSettings,
    Inbox,
    Job,
    Toolbox,
    TurnContext,
    TurnLimits,
    create_line,
    list_rules,
)
from rewind.core.ports import Completion, CompletionRequest, JobPayload, ToolCall

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 10, 28, 9, 40, tzinfo=UTC)
CAPABILITIES = ChannelCapabilities(max_text_chars=4096, max_media_bytes=50_000_000, max_buttons=8)


def _completion(text: str = "", *calls: ToolCall, output_tokens: int = 50) -> Completion:
    charge = Charge(
        provider="meta", model="muse", unit="output_token", quantity=output_tokens,
        usd=Decimal(0),
    )  # fmt: skip
    return Completion(text=text, tool_calls=calls, charges=(charge,))


@dataclass
class ScriptedModel:
    replies: list[Completion]
    requests: list[CompletionRequest] = field(default_factory=list[CompletionRequest])
    model_version: str = "muse-test"

    async def complete(self, request: CompletionRequest) -> Completion:
        self.requests.append(request)
        return self.replies.pop(0) if self.replies else _completion("ok")


@dataclass
class FakeChannel:
    name: str = "telegram"
    sent: list[Reply] = field(default_factory=list[Reply])
    fetch_limit: int = 20_000_000

    @property
    def capabilities(self) -> ChannelCapabilities:
        return CAPABILITIES

    async def send(self, reply: Reply) -> None:
        self.sent.append(reply)

    async def show_progress(self, chat_id: str, key: str, text: str) -> None:
        self.sent.append(Reply(channel=self.name, chat_id=chat_id, text=text))

    async def fetch_attachment(self, attachment: Attachment, destination: Path) -> None:
        if (attachment.size_bytes or 0) > self.fetch_limit:
            raise AttachmentTooLargeError(self.fetch_limit)
        destination.write_bytes(b"video")


class MemoryBlobs:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    async def write(self, key: str, data: bytes) -> None:
        self.data[key] = data

    async def write_file(self, key: str, source: Path) -> None:
        self.data[key] = source.read_bytes()

    async def read(self, key: str) -> bytes:
        return self.data[key]

    async def local_path(self, key: str) -> Path:
        raise NotImplementedError

    async def exists(self, key: str) -> bool:
        return key in self.data

    def signed_url(self, key: str, expires_in_s: int) -> str:
        return f"https://example.test/{key}"


@dataclass
class Setup:
    repository: SqlRepository
    jobs: InProcessJobRunner
    channel: FakeChannel
    model: ScriptedModel
    conversation: Conversation
    inbox: Inbox
    ingested: list[JobPayload]


@pytest.fixture
async def setup(tmp_path: Path) -> AsyncIterator[Setup]:
    repository = SqlRepository.from_url(f"sqlite+aiosqlite:///{tmp_path / 'rewind.db'}")
    await repository.create_tables()
    jobs = InProcessJobRunner()
    channel = FakeChannel()
    model = ScriptedModel([])
    toolbox = Toolbox.of([create_line(repository), list_rules(repository)])
    conversation = Conversation(
        repository=repository,
        store=repository,
        blobs=MemoryBlobs(),
        jobs=jobs,
        channels={"telegram": channel},
        agent=Agent(INSTRUCTIONS, toolbox, TurnLimits(max_tool_calls=2)),
        model_for=lambda workspace_id: model,
        settings=ConversationSettings(frozenset(), "https://tunnel.test"),
        clock=lambda: NOW,
    )
    ingested: list[JobPayload] = []

    async def ingest(payload: JobPayload) -> None:
        ingested.append(payload)

    jobs.register(TURN_JOB, conversation.handle_job)
    jobs.register(INGEST_JOB, ingest)
    yield Setup(repository, jobs, channel, model, conversation, Inbox(repository, jobs), ingested)
    await repository.close()


def _message(
    text: str | None,
    message_id: str = "m1",
    chat_id: str = "42",
    sender_id: str = "7",
    attachments: tuple[Attachment, ...] = (),
) -> CanonicalMessage:
    return CanonicalMessage(
        channel="telegram", message_id=message_id, chat_id=chat_id, sender_id=sender_id,
        received_at=NOW, text=text, attachments=attachments,
    )  # fmt: skip


async def test_a_repeated_delivery_is_answered_once(setup: Setup) -> None:
    setup.model.replies = [_completion("Hi, I'm Rewind.")]

    assert await setup.inbox.receive(_message("hi")) is True
    assert await setup.inbox.receive(_message("hi")) is False
    await setup.jobs.drain()

    assert [reply.text for reply in setup.channel.sent] == ["Hi, I'm Rewind."]
    assert len(setup.model.requests) == 1


async def test_the_agent_calls_a_tool_then_answers_from_its_result(setup: Setup) -> None:
    setup.model.replies = [
        _completion("", ToolCall("c1", "create_line", '{"name": "Packing line 2"}')),
        _completion("Created *Packing line 2*."),
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


async def test_a_turn_stops_offering_tools_after_the_cap(setup: Setup) -> None:
    loop = ToolCall("c", "list_rules", "{}")
    setup.model.replies = [_completion("", loop), _completion("", loop), _completion("Done.")]

    await setup.conversation.handle(_message("list rules"))

    assert [len(request.tools) for request in setup.model.requests] == [2, 2, 0]
    assert setup.channel.sent[-1].text == "Done."


async def test_bad_tool_arguments_go_back_to_the_model(setup: Setup) -> None:
    setup.model.replies = [_completion("", ToolCall("c1", "create_line", '{"title": 3}'))]

    await setup.conversation.handle(_message("make a line"))

    assert "invalid arguments for create_line" in setup.model.requests[1].messages[-1].text


async def test_link_codes_join_a_second_chat_to_the_first_workspace(setup: Setup) -> None:
    await setup.conversation.handle(_message("/link", chat_id="A"))
    code = setup.channel.sent[-1].text.split("*")[1]

    await setup.conversation.handle(_message(f"/link {code}", chat_id="B", message_id="m2"))
    await setup.conversation.handle(_message(f"/link {code}", chat_id="C", message_id="m3"))

    first = await setup.repository.workspace_for_chat(ChatRef("telegram", "A"))
    assert await setup.repository.workspace_for_chat(ChatRef("telegram", "B")) == first
    assert await setup.repository.workspace_for_chat(ChatRef("telegram", "C")) != first
    assert setup.channel.sent[-1].text.startswith("That code is wrong")
    assert setup.model.requests == []


async def test_a_video_is_stored_and_queued_for_ingest(setup: Setup) -> None:
    setup.model.replies = [
        _completion("", ToolCall("c1", "create_line", '{"name": "Packing line 2"}')),
        _completion("Created."),
    ]
    await setup.conversation.handle(_message("New line: Packing line 2"))
    video = Attachment(kind="video", ref="file-1", size_bytes=9_800_000, filename="belt.mp4")

    await setup.conversation.handle(_message(None, message_id="m2", attachments=(video,)))
    await setup.jobs.drain()

    assert (
        setup.ingested[0]["original_key"] == f"footage/{setup.ingested[0]['footage_id']}/belt.mp4"
    )
    assert "Packing line 2" in setup.channel.sent[-1].text


async def test_a_video_over_the_channel_limit_gets_an_upload_link(setup: Setup) -> None:
    setup.model.replies = [
        _completion("", ToolCall("c1", "create_line", '{"name": "Packing line 2"}')),
        _completion("Created."),
    ]
    await setup.conversation.handle(_message("New line: Packing line 2"))
    video = Attachment(kind="video", ref="file-1", size_bytes=612_000_000)

    await setup.conversation.handle(_message(None, message_id="m2", attachments=(video,)))

    assert "https://tunnel.test/u/" in setup.channel.sent[-1].text
    assert setup.ingested == []


async def test_senders_off_the_allowlist_get_no_model_call(setup: Setup, tmp_path: Path) -> None:
    private = Conversation(
        repository=setup.repository, store=setup.repository, blobs=MemoryBlobs(), jobs=setup.jobs,
        channels={"telegram": setup.channel}, agent=Agent(INSTRUCTIONS, Toolbox(), TurnLimits()),
        model_for=lambda workspace_id: setup.model,
        settings=ConversationSettings(frozenset({"telegram:1"}), "https://tunnel.test"),
        clock=lambda: NOW,
    )  # fmt: skip

    await private.handle(_message("hi", sender_id="2"))

    assert setup.channel.sent[-1].text == "Sorry, this assistant is private."
    assert setup.model.requests == []


async def test_a_job_a_turn_starts_runs_only_after_the_turn_has_replied(setup: Setup) -> None:
    async def start(context: TurnContext, argument: str) -> AgentReply:
        return AgentReply("Started.", jobs=(Job("work", {}),))

    async def work(payload: JobPayload) -> None:
        await setup.channel.show_progress("42", "job", "Checking…")

    setup.jobs.register("work", work)
    conversation = Conversation(
        repository=setup.repository, store=setup.repository, blobs=MemoryBlobs(),
        jobs=setup.jobs, channels={"telegram": setup.channel},
        agent=Agent(INSTRUCTIONS, Toolbox.of([]), TurnLimits()),
        model_for=lambda workspace_id: setup.model,
        settings=ConversationSettings(frozenset(), "https://tunnel.test"), clock=lambda: NOW,
        commands={"/go": start},
    )  # fmt: skip

    await conversation.handle(_message("/go"))
    await setup.jobs.drain()

    assert [reply.text for reply in setup.channel.sent] == ["Started.", "Checking…"]
