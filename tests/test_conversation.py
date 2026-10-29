import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rewind.adapters.jobs.in_process import InProcessJobRunner
from rewind.adapters.storage.sql import SqlRepository
from rewind.contracts import Attachment, CanonicalMessage
from rewind.core.conversation import (
    INGEST_JOB,
    INSTRUCTIONS,
    TURN_JOB,
    Agent,
    AgentReply,
    ChatRef,
    Command,
    Conversation,
    ConversationSettings,
    Inbox,
    Job,
    Readings,
    Toolbox,
    TurnContext,
    TurnLimits,
    create_line,
    list_rules,
)
from rewind.core.ports import Completion, CompletionRequest, JobPayload, LanguageModel, ToolCall
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
    ingested: list[JobPayload]
    make: Callable[..., Conversation]


@pytest.fixture
async def setup(tmp_path: Path) -> AsyncIterator[Setup]:
    repository = SqlRepository.from_url(f"sqlite+aiosqlite:///{tmp_path / 'rewind.db'}")
    await repository.create_tables()
    jobs = InProcessJobRunner()
    channel = FakeChannel()
    model = ScriptedModel([])

    def make(
        model: LanguageModel = model,
        toolbox: Toolbox | None = None,
        allowed: frozenset[str] = frozenset(),
        commands: dict[str, Command] | None = None,
        activity_every_s: float = 4.0,
    ) -> Conversation:
        return Conversation(
            repository=repository, store=repository, blobs=FakeBlobStore(), jobs=jobs,
            channels={"telegram": channel},
            agent=Agent(INSTRUCTIONS, toolbox or Toolbox.of([create_line(repository),
                        list_rules(repository)]), TurnLimits(max_tool_calls=2)),
            model_for=lambda workspace_id: model, readings=Readings(),
            settings=ConversationSettings(allowed, "https://tunnel.test",
                                          activity_every_s=activity_every_s),
            clock=lambda: NOW, commands=commands,
        )  # fmt: skip

    conversation = make()
    ingested: list[JobPayload] = []

    async def ingest(payload: JobPayload) -> None:
        ingested.append(payload)

    jobs.register(TURN_JOB, conversation.handle_job)
    jobs.register(INGEST_JOB, ingest)
    inbox = Inbox(repository, jobs)
    yield Setup(repository, jobs, channel, model, conversation, inbox, ingested, make)
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


async def _create_line(setup: Setup) -> None:
    setup.model.replies = [
        completion("", ToolCall("c1", "create_line", '{"name": "Packing line 2"}')),
        completion("Created."),
    ]
    await setup.conversation.handle(_message("New line: Packing line 2"))


# Answers each message with its own text, holding back the ones whose gate is still shut.
@dataclass
class EchoModel:
    gates: dict[str, asyncio.Event] = field(default_factory=dict[str, asyncio.Event])
    delay_s: float = 0.0
    model_version: str = "echo"

    async def complete(self, request: CompletionRequest) -> Completion:
        text = request.messages[-1].text
        if text in self.gates:
            await self.gates[text].wait()
        await asyncio.sleep(self.delay_s)
        return completion(f"re: {text}")


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


async def test_a_turn_stops_offering_tools_after_the_cap(setup: Setup) -> None:
    loop = ToolCall("c", "list_rules", "{}")
    setup.model.replies = [completion("", loop), completion("", loop), completion("Done.")]

    await setup.conversation.handle(_message("list rules"))

    assert [len(request.tools) for request in setup.model.requests] == [2, 2, 0]
    assert setup.channel.sent[-1].text == "Done."


async def test_bad_tool_arguments_go_back_to_the_model(setup: Setup) -> None:
    setup.model.replies = [completion("", ToolCall("c1", "create_line", '{"title": 3}'))]

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


async def test_a_video_is_acknowledged_before_the_download_and_kept_in_history(
    setup: Setup,
) -> None:
    await _create_line(setup)
    video = Attachment(kind="video", ref="file-1", size_bytes=9_800_000, filename="belt.mp4")

    await setup.conversation.handle(_message(None, message_id="m2", attachments=(video,)))
    await setup.jobs.drain()

    [payload] = setup.ingested
    footage_id = str(payload["footage_id"])
    assert payload["original_key"] == f"footage/{footage_id}/belt.mp4"
    assert payload["progress_key"] == footage_id
    # The ack goes out first, as a progress message the ingest job edits later; nothing else.
    shown = [s for s in setup.channel.shown if s.kind != "activity"][1:]
    assert [(s.kind, s.key) for s in shown] == [("progress", footage_id), ("fetch", "")]
    assert shown[0].text.startswith("Got the video for *Packing line 2*.")
    history = await setup.repository.history(ChatRef("telegram", "42"), 10)
    assert [entry.text for entry in history][-2:] == [
        "[sent a video: belt.mp4 for *Packing line 2*]",
        shown[0].text,
    ]


async def test_a_video_over_the_channel_limit_gets_an_upload_link(setup: Setup) -> None:
    await _create_line(setup)
    video = Attachment(kind="video", ref="file-1", size_bytes=612_000_000)

    await setup.conversation.handle(_message(None, message_id="m2", attachments=(video,)))

    acknowledged, refused = setup.channel.sent[-2:]
    assert acknowledged.text.startswith("Got the video")
    assert "https://tunnel.test/u/" in refused.text
    assert setup.ingested == []


async def test_senders_off_the_allowlist_get_no_model_call(setup: Setup) -> None:
    private = setup.make(toolbox=Toolbox(), allowed=frozenset({"telegram:1"}))

    await private.handle(_message("hi", sender_id="2"))

    assert setup.channel.sent[-1].text == "Sorry, this assistant is private."
    assert setup.model.requests == []


async def test_a_job_a_turn_starts_runs_only_after_the_turn_has_replied(setup: Setup) -> None:
    async def start(context: TurnContext, argument: str) -> AgentReply:
        return AgentReply("Started.", jobs=(Job("work", {}),))

    async def work(payload: JobPayload) -> None:
        await setup.channel.show_progress("42", "job", "Checking…")

    setup.jobs.register("work", work)
    conversation = setup.make(toolbox=Toolbox(), commands={"/go": start})

    await conversation.handle(_message("/go"))
    await setup.jobs.drain()

    assert [reply.text for reply in setup.channel.sent] == ["Started.", "Checking…"]


async def test_one_chat_is_answered_in_order_while_other_chats_go_alongside(
    setup: Setup,
) -> None:
    first_gate = asyncio.Event()
    model = EchoModel({"first": first_gate})
    conversation = setup.make(model=model, toolbox=Toolbox())
    setup.jobs.register(TURN_JOB, conversation.handle_job)

    await setup.inbox.receive(_message("first", message_id="1", chat_id="A"))
    await setup.inbox.receive(_message("second", message_id="2", chat_id="A"))
    await setup.inbox.receive(_message("other", message_id="3", chat_id="B"))
    for _ in range(500):
        if any(s.kind == "send" for s in setup.channel.shown):
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.1)
    answered_while_blocked = [s.text for s in setup.channel.shown if s.kind == "send"]
    first_gate.set()
    await setup.jobs.drain()

    assert answered_while_blocked == ["re: other"]
    replies_to_a = [s.text for s in setup.channel.shown if s.kind == "send" and s.chat_id == "A"]
    assert replies_to_a == ["re: first", "re: second"]


async def test_typing_shows_while_a_turn_works_and_stops_with_the_reply(setup: Setup) -> None:
    conversation = setup.make(
        model=EchoModel(delay_s=0.1), toolbox=Toolbox(), activity_every_s=0.02
    )

    await conversation.handle(_message("hi"))
    after_reply = len(setup.channel.shown)
    await asyncio.sleep(0.1)

    kinds = [s.kind for s in setup.channel.shown]
    assert kinds.count("activity") >= 3
    assert kinds[-1] == "send"
    assert len(setup.channel.shown) == after_reply


async def test_a_failing_typing_indicator_does_not_break_the_turn(setup: Setup) -> None:
    setup.channel.activity_fails = True
    conversation = setup.make(model=EchoModel(delay_s=0.05), toolbox=Toolbox(),
                              activity_every_s=0.01)  # fmt: skip

    await conversation.handle(_message("hi"))

    assert setup.channel.sent[-1].text == "re: hi"
