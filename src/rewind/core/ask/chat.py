import asyncio
import logging
import secrets
import tempfile
import uuid
from collections import defaultdict
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rewind.contracts import (
    Attachment,
    Button,
    CanonicalMessage,
    ChannelCapabilities,
    Reply,
    ReplyMedia,
)
from rewind.core.ask.replies import (
    BUSY,
    FETCHING,
    HELP,
    NO_FOOTAGE,
    NOT_ALLOWED,
    NOT_FETCHED,
    NOT_FOOTAGE,
    NOT_HEARD,
    START,
    WORKING,
    answer_text,
    choice_buttons,
    first_look_text,
    progress_text,
    question_buttons,
    tapped,
    timeline_link,
    too_big,
    too_long,
)
from rewind.core.ask.runs import FAILED, SPENT, Runs
from rewind.core.ask.session import (
    Exchange,
    Session,
    Sessions,
    Step,
    footage_suffix,
    is_photo,
)
from rewind.core.ask.voice import AudioTooLongError, Voice
from rewind.core.budget import BudgetExceededError
from rewind.core.ports import (
    ActivityKind,
    AttachmentTooLargeError,
    BlobStore,
    ChannelError,
    ChatChannel,
    ChatRef,
    ChatStore,
    JobPayload,
    JobRunner,
    TranscriberError,
    VideoToolsError,
)

logger = logging.getLogger(__name__)

TURN_JOB = "turn"
# Telegram drops the indicator after about 5 s, so it is sent again a little sooner.
ACTIVITY_EVERY_S = 4.0
# Telegram allows about one edit a second in a chat.
PROGRESS_EVERY_S = 1.0
_FIXED = {"/start": START, "/help": HELP}


class MessageClaims(Protocol):
    async def claim_message(self, channel: str, message_id: str) -> bool: ...


class Inbox:
    def __init__(self, claims: MessageClaims, jobs: JobRunner) -> None:
        self._claims = claims
        self._jobs = jobs

    # A provider retry of the same message is dropped here, before any work is queued.
    async def receive(self, message: CanonicalMessage) -> bool:
        if not await self._claims.claim_message(message.channel, message.message_id):
            return False
        await self._jobs.submit(TURN_JOB, {"message": message.model_dump(mode="json")})
        return True


# Keeps the chat showing that work is going on until stopped. Only a cue: a failure is logged
# and never stops the work it decorates.
class Activity:
    def __init__(
        self,
        channel: ChatChannel,
        chat_id: str,
        every_s: float = ACTIVITY_EVERY_S,
        message_id: str | None = None,
    ) -> None:
        self._channel = channel
        self._chat_id = chat_id
        self._every_s = every_s
        self._message_id = message_id
        self._kind: ActivityKind = "typing"
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "Activity":
        self._task = asyncio.create_task(self._repeat())
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()

    async def show(self, kind: ActivityKind) -> None:
        self._kind = kind
        await self._show()

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        await asyncio.wait([self._task])
        self._task = None

    async def _repeat(self) -> None:
        while True:
            await self._show()
            await asyncio.sleep(self._every_s)

    async def _show(self) -> None:
        try:
            await self._channel.show_activity(self._chat_id, self._kind, self._message_id)
        except ChannelError:
            logger.warning("could not show %s in chat %s", self._kind, self._chat_id, exc_info=True)


# One message showing the steps. A channel that can edit gets them live, at most one edit per
# interval; one that cannot gets the opening line only, so a run is never a stream of messages.
# Only a cue, like Activity: a failure is logged and never stops the paid work behind it.
class Progress:
    def __init__(
        self,
        channel: ChatChannel,
        chat_id: str,
        key: str,
        question: str | None,
        first: str,
        every_s: float,
    ) -> None:
        self._channel = channel
        self._chat_id = chat_id
        self.key = key
        self._question = question
        self._steps = [Step(text=first, done=False)]
        self._every_s = every_s
        self._changed = asyncio.Event()
        self._sending = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "Progress":
        self._changed.set()
        self._task = asyncio.create_task(self._repeat())
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._task is not None:
            # Under the lock: a send cut off halfway could leave a message it cannot find again.
            async with self._sending:
                self._task.cancel()
            await asyncio.wait([self._task])
        if self._changed.is_set():
            await self._show()

    async def step(self, step: Step) -> None:
        self._steps.append(step)
        if self._channel.capabilities.can_edit:
            self._changed.set()

    # What stays above an answer sent as a new message: the steps taken, none still under way.
    async def settle(self) -> None:
        self._steps = [step for step in self._steps if step.done]
        if self._channel.capabilities.can_edit:
            await self._show()

    async def _repeat(self) -> None:
        while True:
            await self._changed.wait()
            async with self._sending:
                await self._show()
            await asyncio.sleep(self._every_s)

    async def _show(self) -> None:
        self._changed.clear()
        text = progress_text(self._question, self._steps)
        try:
            await self._channel.show_progress(self._chat_id, self.key, text)
        except ChannelError:
            logger.warning("could not show progress in chat %s", self._chat_id, exc_info=True)


@dataclass(frozen=True, slots=True)
class ChatSettings:
    # Senders as "channel:sender_id"; empty lets anyone in.
    allowed_senders: frozenset[str]
    public_base_url: str
    activity_every_s: float = ACTIVITY_EVERY_S
    progress_every_s: float = PROGRESS_EVERY_S


@dataclass(frozen=True, slots=True)
class _Turn:
    chat: ChatRef
    channel: ChatChannel
    workspace_id: str
    message_id: str

    @property
    def capabilities(self) -> ChannelCapabilities:
        return self.channel.capabilities


# What stopped a turn, in words for the person.
class _Stopped(Exception):
    def __init__(self, said: str) -> None:
        super().__init__(said)
        self.said = said


# Every chat message comes through here, whatever the channel: footage opens a session, anything
# else is a question about the latest one. Channels differ only in their capabilities.
class Chats:
    def __init__(
        self,
        runs: Runs,
        sessions: Sessions,
        store: ChatStore,
        blobs: BlobStore,
        voice: Voice,
        channels: Mapping[str, ChatChannel],
        settings: ChatSettings,
    ) -> None:
        self._runs = runs
        self._sessions = sessions
        self._store = store
        self._blobs = blobs
        self._voice = voice
        self._channels = channels
        self._settings = settings
        # One chat's messages are answered in the order they came, so a question sent right
        # after a video is about that video; other chats are answered alongside.
        self._turns: defaultdict[ChatRef, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def handle_job(self, payload: JobPayload) -> None:
        await self.handle(CanonicalMessage.model_validate(payload["message"]))

    async def handle(self, message: CanonicalMessage) -> None:
        chat = ChatRef(message.channel, message.chat_id)
        channel = self._channels[chat.channel]
        async with self._turns[chat]:
            if not _allowed(self._settings.allowed_senders, message):
                await _say(channel, chat, NOT_ALLOWED)
                return
            text = (message.text or "").strip()
            fixed = _fixed_reply(text)
            if fixed is not None and not message.attachments:
                await _say(channel, chat, fixed)
                return
            workspace_id = await self._workspace(chat)
            turn = _Turn(chat, channel, workspace_id, message.message_id)
            every_s = self._settings.activity_every_s
            async with Activity(channel, chat.chat_id, every_s, message.message_id) as activity:
                try:
                    await self._take(turn, message.attachments, text, activity)
                except _Stopped as stopped:
                    await _say(channel, chat, stopped.said)

    async def _take(
        self,
        turn: _Turn,
        attachments: tuple[Attachment, ...],
        text: str,
        activity: Activity,
    ) -> None:
        attachment = attachments[0] if attachments else None
        if attachment is not None and attachment.kind != "audio":
            session_id = await self._open(turn, attachment)
            if session_id is not None and text:
                await self._ask(turn, session_id, text, activity)
            return
        picked = tapped(text)
        if picked is None:
            session_id = await self._store.latest_session(turn.chat)
        else:
            session_id = picked[0]
        if session_id is None:
            raise _Stopped(NO_FOOTAGE)
        if attachment is not None:
            text = await self._hear(turn, attachment)
        elif picked is not None:
            text = await self._suggested(session_id, picked[1])
        await self._ask(turn, session_id, text, activity)

    async def _suggested(self, session_id: str, index: int) -> str:
        session = await self._sessions.get(session_id)
        if session is None:
            raise _Stopped(NO_FOOTAGE)
        back = session.exchanges[-1].choices if session.exchanges else ()
        tappable = (*session.first_look.questions, *back)
        if index >= len(tappable):
            raise _Stopped(NO_FOOTAGE)
        return tappable[index]

    # Footage in, first look out. None when it could not be opened and the person has been told;
    # the chat's latest footage then stays as it was.
    async def _open(self, turn: _Turn, attachment: Attachment) -> str | None:
        if attachment.kind == "file":
            raise _Stopped(NOT_FOOTAGE)
        session_id = secrets.token_urlsafe(12)
        chat_id = turn.chat.chat_id
        every_s = self._settings.progress_every_s
        session: Session | None = None
        progress = Progress(turn.channel, chat_id, session_id, None, FETCHING, every_s)
        async with progress:
            try:
                key = await self._fetched(turn, session_id, attachment)
            except _Stopped as stopped:
                problem = stopped.said
            else:
                live = self._runs.open(
                    session_id, key, workspace_id=turn.workspace_id, on_step=progress.step
                )
                await live.finished.wait()
                session = live.session
                problem = live.problem or FAILED
        if session is None:
            await _finish(turn, session_id, problem)
            return None
        await self._store.set_latest_session(turn.chat, session_id)
        buttons = question_buttons(session, turn.capabilities)
        await _finish(turn, session_id, first_look_text(session, turn.capabilities), buttons)
        return session_id

    async def _fetched(self, turn: _Turn, session_id: str, attachment: Attachment) -> str:
        suffix = footage_suffix(attachment.filename, attachment.mime_type)
        if attachment.kind == "image" and not is_photo(f"upload{suffix}"):
            suffix = ".jpg"
        key = f"uploads/{session_id}/upload{suffix}"
        with tempfile.TemporaryDirectory() as workdir:
            local = Path(workdir) / f"upload{suffix}"
            await self._fetch(turn, attachment, local)
            await self._blobs.write_file(key, local)
        return key

    async def _fetch(self, turn: _Turn, attachment: Attachment, destination: Path) -> None:
        try:
            await turn.channel.fetch_attachment(attachment, destination)
        except AttachmentTooLargeError as error:
            base_url = self._settings.public_base_url
            raise _Stopped(too_big(error.limit_bytes, base_url)) from error
        except ChannelError as error:
            logger.exception("could not fetch a file from %s", turn.chat.channel)
            raise _Stopped(NOT_FETCHED) from error

    async def _hear(self, turn: _Turn, attachment: Attachment) -> str:
        with tempfile.TemporaryDirectory() as workdir:
            local = Path(workdir) / "voice"
            await self._fetch(turn, attachment, local)
            try:
                heard = await self._voice.hear(local, turn.workspace_id)
            except BudgetExceededError as error:
                raise _Stopped(SPENT) from error
            except AudioTooLongError as error:
                raise _Stopped(too_long(error.max_seconds)) from error
            except (TranscriberError, VideoToolsError) as error:
                logger.exception("could not transcribe a voice note")
                raise _Stopped(NOT_HEARD) from error
        if not heard:
            raise _Stopped(NOT_HEARD)
        return heard

    async def _ask(self, turn: _Turn, session_id: str, question: str, activity: Activity) -> None:
        key = f"ask-{turn.message_id}"
        every_s = self._settings.progress_every_s
        progress = Progress(turn.channel, turn.chat.chat_id, key, question, WORKING, every_s)
        async with progress:
            live = self._runs.ask(session_id, question, None, turn.workspace_id, progress.step)
            if live is not None:
                await live.finished.wait()
        if live is None:
            await _finish(turn, key, BUSY)
        elif live.exchange is None:
            await _finish(turn, key, live.problem or FAILED)
        else:
            await self._deliver(turn, progress, session_id, live.exchange, activity)

    # One message: the answer with one proof, the rest being on the timeline. The answer still
    # arrives when the proof is too big or will not send.
    async def _deliver(
        self,
        turn: _Turn,
        progress: Progress,
        session_id: str,
        exchange: Exchange,
        activity: Activity,
    ) -> None:
        proof = await self._proof(exchange, turn.capabilities)
        left_out = proof is None and any(f.proof_key for f in exchange.findings)
        link = timeline_link(self._settings.public_base_url, session_id)
        text = answer_text(exchange, link, left_out, turn.capabilities)
        if proof is None:
            session = await self._sessions.get(session_id)
            back = choice_buttons(session, exchange, turn.capabilities) if session else ()
            await _finish(turn, progress.key, text, back)
            return
        await progress.settle()
        await activity.show("upload_video")
        chat = turn.chat
        reply = Reply(
            channel=chat.channel,
            chat_id=chat.chat_id,
            text="",
            media=(proof.model_copy(update={"caption": text}),),
        )
        try:
            await turn.channel.send(reply)
        except ChannelError:
            logger.exception("could not send a proof on %s", chat.channel)
            await _finish(turn, progress.key, text)

    async def _proof(
        self, exchange: Exchange, capabilities: ChannelCapabilities
    ) -> ReplyMedia | None:
        for finding in exchange.findings:
            key = finding.proof_key
            if key is None:
                continue
            kind = "image" if is_photo(key) else "video"
            limit = (
                capabilities.max_image_bytes if kind == "image" else capabilities.max_media_bytes
            )
            if (await self._blobs.local_path(key)).stat().st_size <= limit:
                return ReplyMedia(kind=kind, blob_key=key)
        return None

    async def _workspace(self, chat: ChatRef) -> str:
        workspace_id = await self._store.workspace_for_chat(chat)
        if workspace_id is None:
            workspace_id = uuid.uuid4().hex
            await self._store.bind_chat(chat, workspace_id)
        return workspace_id


# Takes the progress message's place where a channel can edit; otherwise it is a new message.
async def _finish(turn: _Turn, key: str, text: str, buttons: tuple[Button, ...] = ()) -> None:
    chat = turn.chat
    if turn.capabilities.can_edit:
        try:
            await turn.channel.show_progress(chat.chat_id, key, text, buttons)
            return
        except ChannelError:
            logger.warning("could not edit the progress in chat %s", chat.chat_id, exc_info=True)
    reply = Reply(channel=chat.channel, chat_id=chat.chat_id, text=text, buttons=buttons)
    await turn.channel.send(reply)


async def _say(channel: ChatChannel, chat: ChatRef, text: str) -> None:
    await channel.send(Reply(channel=chat.channel, chat_id=chat.chat_id, text=text))


# Fixed answers make no model call, so they are instant and free. Anything else that looks like
# a command gets the help: a stale menu entry must not reach the agent.
def _fixed_reply(text: str) -> str | None:
    if not text.startswith("/"):
        return None
    return _FIXED.get(text.split()[0].partition("@")[0].lower(), HELP)


def _allowed(allowed: Collection[str], message: CanonicalMessage) -> bool:
    return not allowed or f"{message.channel}:{message.sender_id}" in allowed
