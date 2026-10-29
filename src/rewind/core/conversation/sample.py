import uuid
from collections.abc import Callable
from pathlib import Path

from rewind.contracts import Line, ReplyMedia, Rule
from rewind.core.analysis import (
    INGEST_JOB,
    CheckRequest,
    Notifier,
    rule_sentence,
    sample_line_id,
)
from rewind.core.conversation.agent import AgentReply
from rewind.core.conversation.analysis_tools import Analyze
from rewind.core.conversation.readings import Reading, Readings
from rewind.core.conversation.tools import Job, TurnContext
from rewind.core.ports import BlobStore, Repository
from rewind.core.rules import RuleCompiler, normalize

SAMPLE_RULE = "Every box needs a label when it leaves the view."
SAMPLE_TITLE = "the sample"


# [Try a sample]: a real check of a real clip, so someone new sees a result without sending
# anything. It lives on a line of its own; after the first run every frame is cached, so later
# taps are fast and free.
class Sample:
    def __init__(
        self,
        repository: Repository,
        blobs: BlobStore,
        compiler_for: Callable[[str], RuleCompiler],
        readings: Readings,
        notifier: Notifier,
        analyze: Analyze,
        video: Path | None,
    ) -> None:
        self._repository = repository
        self._blobs = blobs
        self._compiler_for = compiler_for
        self._readings = readings
        self._notifier = notifier
        self._analyze = analyze
        self._video = video

    @property
    def available(self) -> bool:
        return self._video is not None

    async def run(self, context: TurnContext, argument: str) -> AgentReply:
        if self._video is None:
            return AgentReply("There's no sample on this bot. Send me a video of your own line.")
        line = await self._line(context)
        rule = await self._rule(context, line)
        caption = (
            f"Sample: a short video of a parcel conveyor.\n"
            f"The rule I'll check: *{rule_sentence(rule.spec)}*."
        )
        chat = context.chat
        footage = await self._repository.line_footage(line.id)
        if footage:
            media = ReplyMedia(kind="video", blob_key=footage[-1].original_key, caption=caption)
            await self._notifier.notify(chat.channel, chat.chat_id, "", media=(media,))
            offer = await self._analyze.offer(
                context, line, footage[-1:], (rule,), CheckRequest(title=SAMPLE_TITLE),
                context.reply_key,
            )  # fmt: skip
            return AgentReply(offer.reply, offer.buttons, jobs=offer.jobs)
        footage_id = uuid.uuid4().hex
        original_key = f"footage/{footage_id}/{self._video.name}"
        await self._blobs.write_file(original_key, self._video)
        media = ReplyMedia(kind="video", blob_key=original_key, caption=caption)
        await self._notifier.notify(chat.channel, chat.chat_id, "", media=(media,))
        self._readings.start(Reading(footage_id, line.id, self._video.name))
        await self._notifier.progress(
            chat.channel, chat.chat_id, context.reply_key, "✓ Video received\n⏳ Reading it…"
        )
        payload = {
            "workspace_id": context.workspace_id,
            "line_id": line.id,
            "footage_id": footage_id,
            "original_key": original_key,
            "channel": chat.channel,
            "chat_id": chat.chat_id,
            "progress_key": context.reply_key,
            "title": SAMPLE_TITLE,
        }
        return AgentReply("", jobs=(Job(INGEST_JOB, payload),))

    async def _line(self, context: TurnContext) -> Line:
        line_id = sample_line_id(context.workspace_id)
        line = await self._repository.get_line(line_id)
        if line is None:
            line = Line(
                id=line_id, workspace_id=context.workspace_id, name="Sample",
                created_at=context.now,
            )  # fmt: skip
            await self._repository.save_line(line)
        return line

    async def _rule(self, context: TurnContext, line: Line) -> Rule:
        normalized = normalize(SAMPLE_RULE)
        for rule in await self._repository.rules(line.id):
            if rule.normalized_text == normalized and rule.active:
                return rule
        # The compile cache makes this free after the first sample on this bot.
        compiled = await self._compiler_for(context.workspace_id).compile(SAMPLE_RULE)
        rule = Rule(
            id=uuid.uuid4().hex, line_id=line.id, text=SAMPLE_RULE, normalized_text=normalized,
            spec=compiled.spec, created_at=context.now,
        )  # fmt: skip
        await self._repository.save_rule(rule)
        return rule
