import logging
import re
import secrets
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Collection, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from rewind.contracts import Attachment, CanonicalMessage, Reply
from rewind.core.analysis import INGEST_JOB
from rewind.core.budget import BudgetExceededError
from rewind.core.conversation.agent import Agent, AgentReply
from rewind.core.conversation.channel import ChatChannel
from rewind.core.conversation.prompt import workspace_state
from rewind.core.conversation.store import ChatRef, ConversationStore, HistoryEntry
from rewind.core.conversation.tools import ToolError, TurnContext
from rewind.core.conversation.workspace_tools import UPLOAD_TOKEN, make_upload_link, resolve_line
from rewind.core.ports import (
    BlobStore,
    ChannelError,
    JobPayload,
    JobRunner,
    LanguageModel,
    LanguageModelError,
    Repository,
)

logger = logging.getLogger(__name__)

TURN_JOB = "turn"
LINK_TOKEN = "link"
HISTORY_TURNS = 20
NOT_ALLOWED = "Sorry, this assistant is private."
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

Command = Callable[[TurnContext, str], Awaitable[AgentReply]]


class AttachmentTooLargeError(ChannelError):
    """Raised by a channel when the provider will not hand over a file this big."""

    def __init__(self, limit_bytes: int) -> None:
        super().__init__(f"the channel can fetch files up to {limit_bytes} bytes")
        self.limit_bytes = limit_bytes


@dataclass(frozen=True, slots=True)
class ConversationSettings:
    # Senders as "channel:sender_id"; empty lets anyone in.
    allowed_senders: frozenset[str]
    public_base_url: str
    upload_ttl: timedelta = timedelta(hours=24)
    link_ttl: timedelta = timedelta(minutes=10)


class Inbox:
    def __init__(self, repository: Repository, jobs: JobRunner) -> None:
        self._repository = repository
        self._jobs = jobs

    # D-7: a provider retry of the same message is dropped here, before any work is queued.
    async def receive(self, message: CanonicalMessage) -> bool:
        if not await self._repository.claim_message(message.channel, message.message_id):
            return False
        await self._jobs.submit(TURN_JOB, {"message": message.model_dump(mode="json")})
        return True


class Conversation:
    def __init__(
        self,
        repository: Repository,
        store: ConversationStore,
        blobs: BlobStore,
        jobs: JobRunner,
        channels: Mapping[str, ChatChannel],
        agent: Agent,
        model_for: Callable[[str], LanguageModel],
        settings: ConversationSettings,
        clock: Callable[[], datetime],
        commands: Mapping[str, Command] | None = None,
    ) -> None:
        self._repository = repository
        self._store = store
        self._blobs = blobs
        self._jobs = jobs
        self._channels = channels
        self._agent = agent
        self._model_for = model_for
        self._settings = settings
        self._clock = clock
        self._commands: dict[str, Command] = {"/link": self._link, **(commands or {})}

    async def handle_job(self, payload: JobPayload) -> None:
        await self.handle(CanonicalMessage.model_validate(payload["message"]))

    async def handle(self, message: CanonicalMessage) -> None:
        chat = ChatRef(message.channel, message.chat_id)
        if not _allowed(self._settings.allowed_senders, message):
            await self.send(chat, AgentReply(NOT_ALLOWED))
            return
        context = TurnContext(await self._workspace(chat), chat, self._clock())
        try:
            reply = await self._answer(context, message)
        except LanguageModelError:
            logger.exception("language model failed for %s", chat)
            reply = AgentReply(
                "I couldn't reach the language model just now. Try again in a minute."
            )
        except BudgetExceededError as error:
            reply = AgentReply(f"I've stopped making paid calls: {error}.")
        await self.send(chat, reply)

    # The line name an upload link is for, without using the link up.
    async def upload_target(self, token: str) -> str | None:
        payload = await self._store.peek_token(UPLOAD_TOKEN, token, self._clock())
        if payload is None:
            return None
        line = await self._repository.get_line(str(payload["line_id"]))
        return None if line is None else line.name

    async def receive_upload(self, token: str, filename: str, source: Path) -> bool:
        payload = await self._store.take_token(UPLOAD_TOKEN, token, self._clock())
        if payload is None:
            return False
        footage_id = uuid.uuid4().hex
        original_key = f"footage/{footage_id}/{_safe_name(filename, self._clock())}"
        await self._blobs.write_file(original_key, source)
        await self._jobs.submit(
            INGEST_JOB, {**payload, "footage_id": footage_id, "original_key": original_key}
        )
        chat = ChatRef(str(payload["channel"]), str(payload["chat_id"]))
        await self.send(chat, AgentReply(f"Upload finished: *{filename}*. I'm reading it now."))
        return True

    async def send(self, chat: ChatRef, reply: AgentReply) -> None:
        channel = self._channels[chat.channel]
        await channel.send(
            Reply(
                channel=chat.channel, chat_id=chat.chat_id, text=reply.text,
                media=reply.media, buttons=reply.buttons,
            )
        )  # fmt: skip

    async def _answer(self, context: TurnContext, message: CanonicalMessage) -> AgentReply:
        text = (message.text or "").strip()
        command, _, argument = text.partition(" ")
        handler = self._commands.get(command.lower())
        if handler is not None:
            return await handler(context, argument.strip())
        notes: list[str] = []
        for attachment in message.attachments:
            notes.append(await self._receive_attachment(context, message.channel, attachment))
        if not text:
            return AgentReply("\n\n".join(notes))
        if notes:
            text = "\n".join(f"[{note}]" for note in notes) + "\n" + text
        return await self._turn(context, text)

    async def _turn(self, context: TurnContext, text: str) -> AgentReply:
        lines = await self._repository.lines(context.workspace_id)
        rules = {
            line.id: tuple(r for r in await self._repository.rules(line.id) if r.active)
            for line in lines
        }
        footage = {line.id: await self._repository.line_footage(line.id) for line in lines}
        state = workspace_state(sorted(lines, key=lambda line: line.created_at), rules, footage)
        history = await self._store.history(context.chat, HISTORY_TURNS)
        reply = await self._agent.run(
            self._model_for(context.workspace_id), context, state, history, text
        )
        await self._store.append_history(context.chat, HistoryEntry("user", text), context.now)
        await self._store.append_history(
            context.chat, HistoryEntry("assistant", reply.text), self._clock()
        )
        return reply

    async def _receive_attachment(
        self, context: TurnContext, channel_name: str, attachment: Attachment
    ) -> str:
        if attachment.kind != "video":
            return "I can only check videos. Send the footage as a video file."
        try:
            line = await resolve_line(self._repository, context.workspace_id, None)
        except ToolError:
            return "Tell me the name of the line first, then send the footage again."
        footage_id = uuid.uuid4().hex
        name = _safe_name(attachment.filename, context.now)
        original_key = f"footage/{footage_id}/{name}"
        try:
            with tempfile.TemporaryDirectory() as workdir:
                local = Path(workdir) / name
                await self._channels[channel_name].fetch_attachment(attachment, local)
                await self._blobs.write_file(original_key, local)
        except AttachmentTooLargeError as error:
            url = await make_upload_link(
                self._store, self._settings.public_base_url, self._settings.upload_ttl,
                context, line,
            )  # fmt: skip
            hours = int(self._settings.upload_ttl.total_seconds() // 3600)
            return (
                f"That file is too big for me to fetch here (the limit is "
                f"{error.limit_bytes // 1_000_000} MB). Upload it with this link instead:\n"
                f"{url}\n\n"
                f"It works once, for {hours} hours, and goes straight to *{line.name}*."
            )
        await self._jobs.submit(
            INGEST_JOB,
            {
                "workspace_id": context.workspace_id,
                "line_id": line.id,
                "footage_id": footage_id,
                "original_key": original_key,
                "channel": context.chat.channel,
                "chat_id": context.chat.chat_id,
            },
        )
        return (
            f"Got the video for *{line.name}*. I'm reading it now and will tell you when "
            "it's ready."
        )

    async def _workspace(self, chat: ChatRef) -> str:
        workspace_id = await self._store.workspace_for_chat(chat)
        if workspace_id is None:
            workspace_id = uuid.uuid4().hex
            await self._store.bind_chat(chat, workspace_id)
        return workspace_id

    async def _link(self, context: TurnContext, argument: str) -> AgentReply:
        if not argument:
            code = f"{secrets.randbelow(10**6):06d}"
            await self._store.put_token(
                LINK_TOKEN, code, {"workspace_id": context.workspace_id},
                context.now + self._settings.link_ttl,
            )  # fmt: skip
            minutes = int(self._settings.link_ttl.total_seconds() // 60)
            return AgentReply(
                f"Your link code is *{code}*.\nSend /link {code} to Rewind on your other chat app "
                f"within {minutes} minutes. Both chats will then share the same lines, rules "
                "and results."
            )
        payload = await self._store.take_token(LINK_TOKEN, argument.replace(" ", ""), context.now)
        if payload is None:
            return AgentReply(
                "That code is wrong or has expired. Send /link on the other chat for a new one."
            )
        workspace_id = str(payload["workspace_id"])
        await self._store.bind_chat(context.chat, workspace_id)
        lines = await self._repository.lines(workspace_id)
        return AgentReply(f"Linked. This chat now shares a workspace with {len(lines)} lines.")


# The file name is kept in the blob key: it is how people name a recording in chat.
def _safe_name(filename: str | None, now: datetime) -> str:
    name = _UNSAFE_NAME.sub("_", Path(filename or "").name).strip("._")
    if not name:
        return f"clip_{now:%m%d_%H%M%S}.mp4"
    return name if Path(name).suffix else f"{name}.mp4"


def _allowed(allowed: Collection[str], message: CanonicalMessage) -> bool:
    return not allowed or f"{message.channel}:{message.sender_id}" in allowed
