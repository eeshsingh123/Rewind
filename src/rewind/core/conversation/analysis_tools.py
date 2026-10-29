import logging
import secrets
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Literal

from rewind.contracts import (
    Button,
    ChannelCapabilities,
    CountAtLeast,
    EndOfFootage,
    EntersZone,
    ExitsView,
    Footage,
    Has,
    InZone,
    Line,
    Rule,
    RuleSpec,
    TimeSpan,
)
from rewind.core.analysis import (
    ANALYZE_JOB,
    Analysis,
    Checklist,
    CheckRequest,
    Clip,
    Notifier,
    Read,
    about_time,
    check_header,
    check_payload,
    check_request,
    check_steps,
    clip_buttons,
    duration_text,
    footage_name,
    is_sample_line,
    rule_sentence,
    video_length,
    violation_line,
)
from rewind.core.budget import BudgetExceededError
from rewind.core.conversation.agent import AgentReply
from rewind.core.conversation.readings import Reading, Readings, WaitingCheck
from rewind.core.conversation.store import ChatRef, ConversationStore
from rewind.core.conversation.tools import (
    Job,
    ToolArguments,
    ToolError,
    ToolResult,
    ToolSpec,
    TurnContext,
)
from rewind.core.conversation.workspace_tools import resolve_line
from rewind.core.ports import JobPayload, JobRunner, Repository, VideoToolsError
from rewind.core.rules import RuleCompileError, RuleCompiler, normalize, zone_keys

logger = logging.getLogger(__name__)

CONFIRM_TOKEN = "analyze"
CONFIRM_TTL = timedelta(hours=1)
ALL = "all"
RULE_EXAMPLE = '"every box has a label when it leaves the view"'
SEND_VIDEO = "Now send me a video of the line."


@dataclass(frozen=True, slots=True)
class AnalyzeSettings:
    # An estimate above this waits for a tap; the daily cap is the ledger's.
    confirm_above_usd: Decimal
    daily_cap_usd: Decimal


class AddRuleArguments(ToolArguments):
    text: str
    line: str | None = None


class RecordingArguments(ToolArguments):
    line: str | None = None
    # A recording's file name, or "all"; the newest recording when left out.
    recording: str | None = None


class AskArguments(ToolArguments):
    question: str
    line: str | None = None
    recording: str | None = None
    # A stretch of the recording the question is about, in seconds from its start.
    from_s: float | None = None
    to_s: float | None = None


class ClipArguments(ToolArguments):
    number: int
    line: str | None = None
    recording: str | None = None


async def resolve_footage(
    repository: Repository, line: Line, name: str | None
) -> tuple[Footage, ...]:
    footage = await repository.line_footage(line.id)
    if not footage:
        raise ToolError(f"line {line.name!r} has no recordings yet; send a video first")
    if name is None:
        return (footage[-1],)
    if name.strip().casefold() == ALL:
        return footage
    wanted = name.strip().casefold()
    matches = tuple(f for f in footage if wanted in footage_name(f).casefold())
    if not matches:
        known = ", ".join(footage_name(f) for f in footage)
        raise ToolError(f"no recording named {name!r}; the recordings are: {known}")
    return matches[-1:]


def describe_rule(spec: RuleSpec, line: Line, zone_phrases: Mapping[str, str]) -> str:
    def part(key: str) -> str:
        return spec.parts.get(key, key)

    needs: list[str] = []
    for requirement in spec.requirements:
        match requirement.predicate:
            case Has(part=key):
                text = f"has {part(key)}"
            case CountAtLeast(part=key, n=n):
                text = f"has at least {n} {part(key)}"
            case InZone(zone=zone):
                text = f"is in {zone}"
        if requirement.timing == "ever_before":
            text += " at some point before"
        needs.append(text)
    needs += [f"{part(step.first)} before {part(step.then)}" for step in spec.order]
    match spec.checkpoint:
        case EntersZone(zone=zone):
            when = f"when it enters {zone}"
        case ExitsView():
            when = "when it leaves the view"
        case EndOfFootage():
            when = "at the end of the recording"
    lines = [f"item: {spec.entity.phrase}"]
    if needs:
        lines.append(f"must: {'; '.join(needs)}")
    lines.append(f"checked {when}")
    if spec.dwell is not None:
        lines.append(f"may stay in {spec.dwell.zone} at most {spec.dwell.max_s:g} s")
    if spec.rate is not None:
        lines.append(f"at least {spec.rate.min_per_minute:g} per minute")
    drawn = {zone.key for zone in line.zones}
    for key in sorted(zone_keys(spec) - drawn):
        phrase = zone_phrases.get(key, key.replace("_", " "))
        lines.append(f"area {key!r} is not marked on this line; I will look for {phrase!r}")
    return "\n".join(lines)


# The read-back: what the rule will check, in one plain sentence, before it is saved.
def add_rule(
    repository: Repository, compiler_for: Callable[[str], RuleCompiler]
) -> ToolSpec[AddRuleArguments]:
    async def run(context: TurnContext, arguments: AddRuleArguments) -> ToolResult:
        line = await resolve_line(repository, context.workspace_id, arguments.line)
        normalized = normalize(arguments.text)
        active = [rule for rule in await repository.rules(line.id) if rule.active]
        for index, rule in enumerate(active, 1):
            if rule.normalized_text == normalized:
                said = f"You already have that rule: rule {index}. Send a video to check it."
                return ToolResult(f"line {line.name!r} already has this rule", reply=said)
        try:
            compiled = await compiler_for(context.workspace_id).compile(arguments.text)
        except RuleCompileError as error:
            raise ToolError(f"I could not turn that into a rule: {error}") from error
        except BudgetExceededError as error:
            raise ToolError(f"the spending limit stops me: {error}") from error
        spec = compiled.spec
        # Saved as a draft: it counts only once the person says the reading is right.
        rule = Rule(
            id=uuid.uuid4().hex, line_id=line.id, text=arguments.text.strip(),
            normalized_text=normalized, spec=spec, active=False, created_at=context.now,
        )  # fmt: skip
        await repository.save_rule(rule)
        facts = describe_rule(spec, line, compiled.zone_phrases)
        sentence = rule_sentence(spec)
        missing = sorted(zone_keys(spec) - {zone.key for zone in line.zones})
        found_by_text = "".join(
            f"\nI'll look for the {compiled.zone_phrases.get(key, key.replace('_', ' '))} in "
            "the video, as it isn't marked on this line."
            for key in missing
        )
        if not spec.unsupported:
            said = f"I'll check: *{sentence}*.{found_by_text}" if sentence else ""
            return ToolResult(
                f"read back to the person, waiting for their OK:\n{facts}",
                buttons=(
                    Button(label="Looks right", value=f"/keep_rule {rule.id}"),
                    Button(label="Change", value=f"/drop_rule {rule.id}"),
                ),
                reply=said or "I couldn't find anything to check in that rule. Say it again?",
            )
        dropped = "; ".join(f'"{clause}"' for clause in spec.unsupported)
        if not sentence:
            said = f"I can't check {dropped} yet. Try a rule about what each item must have."
            return ToolResult(f"nothing in it can be checked: {dropped}", reply=said)
        said = (
            f"I can check this part: *{sentence}*.\nI can't check: {dropped}.{found_by_text}\n\n"
            "Save the part I can check?"
        )
        return ToolResult(
            f"part of it cannot be checked ({dropped}); asked whether to save the rest:\n{facts}",
            buttons=(
                Button(label="Save that part", value=f"/keep_rule {rule.id}"),
                Button(label="Don't save", value=f"/drop_rule {rule.id}"),
            ),
            reply=said,
        )

    return ToolSpec(
        "add_rule",
        "Add a rule to a line. Pass the person's sentence exactly as written. The tool reads "
        "the rule back to the person itself.",
        AddRuleArguments,
        run,
    )


@dataclass(frozen=True, slots=True)
class Offer:
    # Facts for the model's history.
    facts: tuple[str, ...]
    total_usd: Decimal
    outcome: Literal["started", "confirm", "over_limit"]
    # What the person is told; empty when the check started, as its checklist says so.
    reply: str = ""
    buttons: tuple[Button, ...] = ()
    jobs: tuple[Job, ...] = ()


class Analyze:
    def __init__(
        self,
        repository: Repository,
        store: ConversationStore,
        analysis: Analysis,
        readings: Readings,
        notifier: Notifier,
        jobs: JobRunner,
        settings: AnalyzeSettings,
        clock: Callable[[], datetime],
    ) -> None:
        self._repository = repository
        self._store = store
        self._analysis = analysis
        self._readings = readings
        self._notifier = notifier
        self._jobs = jobs
        self._settings = settings
        self._clock = clock

    def tool(self) -> ToolSpec[RecordingArguments]:
        return ToolSpec(
            "analyze",
            "Check recordings of a line against its rules: starts the check, or offers a "
            "confirm button when it costs more than the threshold. The tool tells the person "
            "itself.",
            RecordingArguments,
            self._run,
        )

    async def confirm(self, context: TurnContext, argument: str) -> AgentReply:
        payload = await self._store.take_token(CONFIRM_TOKEN, argument, context.now)
        if payload is None:
            return AgentReply("That button has expired or was used already. Ask me again.")
        payload = {**payload, "progress_key": context.reply_key}
        await self._show_checklist(context.chat, payload)
        return AgentReply("", jobs=(Job(ANALYZE_JOB, payload),))

    async def cancel(self, context: TurnContext, argument: str) -> AgentReply:
        if argument:
            await self._store.take_token(CONFIRM_TOKEN, argument, context.now)
        return AgentReply("OK, I won't check it. Send a shorter video whenever you like.")

    # [Looks right] on a rule read back: the rule counts from now, and a video already sent is
    # checked against it at once.
    async def keep_rule(self, context: TurnContext, argument: str) -> AgentReply:
        rule = await self._repository.get_rule(argument)
        line = None if rule is None else await self._repository.get_line(rule.line_id)
        if rule is None or line is None:
            return AgentReply("I can't find that rule any more. Send it again?")
        if not rule.active:
            await self._repository.save_rule(rule.model_copy(update={"active": True}))
        rules = tuple(r for r in await self._repository.rules(line.id) if r.active)
        number = next(i for i, r in enumerate(rules, 1) if r.id == rule.id)
        lead = f"Saved as rule {number}. "
        reading = self._readings.of_line(line.id)
        if reading:
            waiting = WaitingCheck(context.workspace_id, context.chat, None)
            self._readings.wait(reading[-1].footage_id, waiting)
            return AgentReply(f"{lead}I'll check your video as soon as I've read it.")
        footage = await self._repository.line_footage(line.id)
        if not footage:
            return AgentReply(f"{lead}{SEND_VIDEO}")
        offer = await self.offer(
            context, line, footage[-1:], rules, CheckRequest(lead=lead), context.reply_key
        )
        return AgentReply(offer.reply, offer.buttons, jobs=offer.jobs)

    async def drop_rule(self, context: TurnContext, argument: str) -> AgentReply:
        return AgentReply("OK, I didn't save it. Tell me the rule again in other words.")

    # [Check older footage]: every recording of the line, against every rule.
    async def older(self, context: TurnContext, argument: str) -> AgentReply:
        line = await self._repository.get_line(argument)
        if line is None or line.workspace_id != context.workspace_id:
            return AgentReply("I can't find that line any more.")
        rules = tuple(r for r in await self._repository.rules(line.id) if r.active)
        footage = await self._repository.line_footage(line.id)
        if not rules or not footage:
            return AgentReply("There is nothing to check yet: this line needs a rule and a video.")
        offer = await self.offer(context, line, footage, rules, CheckRequest(), context.reply_key)
        return AgentReply(offer.reply, offer.buttons, jobs=offer.jobs)

    # The ingest job. Its result replaces the "reading it" message. When the line has rules, the
    # same message turns into the check's checklist, under the same cost rule as any check.
    async def handle_ingest(self, payload: JobPayload) -> None:
        channel, chat_id = str(payload["channel"]), str(payload["chat_id"])
        key = str(payload["progress_key"])
        failed = False
        try:
            read = await self._analysis.read(payload)
        except VideoToolsError:
            logger.exception("ingest failed")
            read, failed = None, True
        finally:
            waiting = self._readings.finish(str(payload["footage_id"]))
        if failed:
            text = "I couldn't read that video. Is it a normal MP4 or MOV file?"
            await self._notifier.progress(channel, chat_id, key, text)
            return
        if read is None:
            return
        chat = waiting.chat if waiting is not None else ChatRef(channel, chat_id)
        workspace_id = waiting.workspace_id if waiting else str(payload["workspace_id"])
        await self._check_new_video(read, chat, workspace_id, waiting, payload, key)

    async def _check_new_video(
        self,
        read: Read,
        chat: ChatRef,
        workspace_id: str,
        waiting: WaitingCheck | None,
        payload: JobPayload,
        key: str,
    ) -> None:
        rules = tuple(r for r in await self._repository.rules(read.line.id) if r.active)
        if not rules:
            text = f"{read.summary}\n\n{read.next_step}"
            await self._notifier.progress(chat.channel, chat.chat_id, key, text)
            return
        footage: tuple[Footage, ...] = (read.footage,)
        if waiting is not None and (waiting.recording or "").strip().casefold() == ALL:
            footage = await self._repository.line_footage(read.line.id)
        context = TurnContext(workspace_id, chat, self._clock(), key)
        title = payload.get("title")
        request = CheckRequest(title=str(title) if title else None)
        offer = await self.offer(context, read.line, footage, rules, request, key)
        if offer.outcome != "started":
            text = f"{read.summary}\n\n{offer.reply}"
            await self._notifier.progress(chat.channel, chat.chat_id, key, text, offer.buttons)
        for job in offer.jobs:
            await self._jobs.submit(job.kind, job.payload)

    async def _run(self, context: TurnContext, arguments: RecordingArguments) -> ToolResult:
        line = await resolve_line(self._repository, context.workspace_id, arguments.line)
        rules = tuple(r for r in await self._repository.rules(line.id) if r.active)
        if not rules:
            said = f"*{line.name}* has no rules yet. Tell me one in plain words, for example: "
            return ToolResult(f"line {line.name!r} has no rules", reply=said + RULE_EXAMPLE + ".")
        reading = _reading_asked_for(self._readings.of_line(line.id), arguments.recording)
        if reading is not None:
            waiting = WaitingCheck(context.workspace_id, context.chat, arguments.recording)
            self._readings.wait(reading.footage_id, waiting)
            return ToolResult(
                f"NOT started yet: {reading.name} is still being read; it starts by itself",
                reply="I'm still reading that video. I'll check it as soon as it's ready.",
            )
        footage = await resolve_footage(self._repository, line, arguments.recording)
        offer = await self.offer(context, line, footage, rules, CheckRequest(), context.reply_key)
        return ToolResult(
            "\n".join(offer.facts), buttons=offer.buttons, jobs=offer.jobs, reply=offer.reply
        )

    async def offer(
        self,
        context: TurnContext,
        line: Line,
        footage: Sequence[Footage],
        rules: Sequence[Rule],
        request: CheckRequest,
        key: str,
    ) -> Offer:
        estimate = await self._analysis.estimate(context.workspace_id, line, footage, rules)
        start_of_day = context.now.replace(hour=0, minute=0, second=0, microsecond=0)
        spent = await self._repository.spent_usd(start_of_day, context.workspace_id)
        total = estimate.total_usd
        facts = [
            f"{footage_name(f.footage)}: {f.frames} frames to look at, {f.cached_frames} "
            f"already seen (free), about ${f.usd:.2f}"
            for f in estimate.footage
        ]
        if estimate.parts_on_stills:
            facts.append(
                f"plus about ${estimate.per_item_usd:.4f} per item, for one photo of each item "
                f"to look for: {', '.join(estimate.parts_on_stills)}"
            )
        if estimate.zones_to_find:
            facts.append(
                f"finding {', '.join(estimate.zones_to_find)} in the video once: "
                f"${estimate.zones_usd:.3f}"
            )
        facts.append(f"total about ${total:.2f}, about {about_time(estimate.seconds)}")
        facts.append(f"spent today ${spent:.2f} of the ${self._settings.daily_cap_usd:.2f} limit")
        request = CheckRequest(request.title, request.lead, estimate.seconds, total)
        payload: JobPayload = {
            "workspace_id": context.workspace_id,
            "line_id": line.id,
            "footage_ids": [f.id for f in footage],
            "rule_ids": [r.id for r in rules],
            "channel": context.chat.channel,
            "chat_id": context.chat.chat_id,
            **check_payload(request),
        }
        what = _what(footage)
        if spent + total > self._settings.daily_cap_usd:
            facts.append("NOT started: it would pass today's limit")
            said = (
                f"{request.lead}Checking {what} would cost about ${total:.2f}, which passes "
                f"today's limit of ${self._settings.daily_cap_usd:.2f}, so I didn't start it."
            )
            return Offer(tuple(facts), total, "over_limit", said)
        if total <= self._settings.confirm_above_usd:
            facts.append("started at once; a checklist in the chat shows how far it is")
            payload["progress_key"] = key
            await self._show_checklist(context.chat, payload)
            return Offer(tuple(facts), total, "started", jobs=(Job(ANALYZE_JOB, payload),))
        token = secrets.token_urlsafe(12)
        await self._store.put_token(CONFIRM_TOKEN, token, dict(payload), context.now + CONFIRM_TTL)
        facts.append("NOT started yet: waiting for the person to tap the confirm button")
        said = (
            f"{request.lead}Checking {what} takes {about_time(estimate.seconds)} and costs from "
            f"*${total:.2f}*.\nYou've spent ${spent:.2f} of today's "
            f"${self._settings.daily_cap_usd:.2f}."
        )
        buttons = (
            Button(label=f"Check it (${total:.2f})", value=f"/confirm {token}"),
            Button(label="Not now", value=f"/cancel {token}"),
        )
        return Offer(tuple(facts), total, "confirm", said, buttons)

    # The checklist goes up the moment a check is started, so it shows even while another
    # check is still running ahead of it.
    async def _show_checklist(self, chat: ChatRef, payload: JobPayload) -> None:
        footage_ids = payload["footage_ids"]
        rule_ids = payload["rule_ids"]
        if not isinstance(footage_ids, list) or not isinstance(rule_ids, list) or not footage_ids:
            return
        footage = await self._repository.get_footage(str(footage_ids[0]))
        if footage is None:
            return
        rules = [r for r in await self._repository.rules(footage.line_id) if r.id in rule_ids]
        request = check_request(payload)
        if len(footage_ids) > 1:
            what = f"*{footage_name(footage)}* (1 of {len(footage_ids)})"
        else:
            what = request.title or "your video"
        checklist = Checklist(
            self._notifier, chat.channel, chat.chat_id, str(payload["progress_key"]),
            check_header(request, what), check_steps(footage, rules),
        )  # fmt: skip
        await checklist.show()


def _what(footage: Sequence[Footage]) -> str:
    if len(footage) == 1:
        return f"*{footage_name(footage[0])}* ({video_length(footage[0].duration_s)})"
    total = sum(f.duration_s for f in footage)
    return f"{len(footage)} videos ({duration_text(total)} in all)"


# The recording a check asks for, if it is one still being read: the newest by default.
def _reading_asked_for(reading: Sequence[Reading], recording: str | None) -> Reading | None:
    if not reading:
        return None
    wanted = (recording or "").strip().casefold()
    if not wanted or wanted == ALL:
        return reading[-1]
    matches = [r for r in reading if wanted in r.name.casefold()]
    return matches[-1] if matches else None


class Findings:
    def __init__(
        self,
        repository: Repository,
        analysis: Analysis,
        capabilities: Mapping[str, ChannelCapabilities],
    ) -> None:
        self._repository = repository
        self._analysis = analysis
        self._capabilities = capabilities

    def violations_tool(self) -> ToolSpec[RecordingArguments]:
        return ToolSpec(
            "violations",
            "List the violations found in recordings of a line, numbered per recording.",
            RecordingArguments,
            self._violations,
        )

    def clip_tool(self) -> ToolSpec[ClipArguments]:
        return ToolSpec(
            "get_clip",
            "Send the proof clip of one violation, by its number in a recording's list.",
            ClipArguments,
            self._clip,
        )

    def ask_tool(self) -> ToolSpec[AskArguments]:
        return ToolSpec(
            "ask_footage",
            "Answer a question about what happened in a checked recording: when failures started, "
            "how often, patterns, stops, pace, a stretch of time, or how it compares with the "
            "line's other recordings. Pass the person's question as written; pass from_s and to_s "
            "when it names a stretch of time. The answer is sent as it is.",
            AskArguments,
            self._ask,
        )

    # /insight <footage id> <insight id>: the clip behind one line of a result or an answer.
    async def insight_command(self, context: TurnContext, argument: str) -> AgentReply:
        footage_id, _, insight_id = argument.partition(" ")
        footage = await self._repository.get_footage(footage_id)
        insight = None if footage is None else await self._analysis.insight(footage, insight_id)
        if footage is None or insight is None:
            return AgentReply("I can't find that clip any more.")
        max_bytes = self._capabilities[context.chat.channel].max_media_bytes
        clip = await self._analysis.insight_clip(footage, insight, max_bytes)
        return AgentReply("" if clip.media else clip.caption, (), clip.media)

    async def clip_command(self, context: TurnContext, argument: str) -> AgentReply:
        footage_id, _, number = argument.partition(" ")
        footage = await self._repository.get_footage(footage_id)
        if footage is None or not number.isdigit():
            return AgentReply("I can't find that clip any more.")
        clip, buttons = await self._make_clip(context, footage, int(number))
        return AgentReply("" if clip.media else clip.caption, buttons, clip.media)

    # /clips: the clips of the newest checked video that has findings, your own line first.
    async def clips_command(self, context: TurnContext, argument: str) -> AgentReply:
        lines = sorted(await self._repository.lines(context.workspace_id),
                       key=lambda line: (is_sample_line(line), line.created_at))  # fmt: skip
        own = [line for line in lines if not is_sample_line(line)]
        ordered = list(reversed(own)) + [line for line in lines if is_sample_line(line)]
        for line in ordered:
            for footage in reversed(await self._repository.line_footage(line.id)):
                if await self._repository.violations(footage.id):
                    return await self.clip_command(context, f"{footage.id} 1")
        return AgentReply(
            "No clips yet: no check has found a problem so far. Send a video and a rule, "
            "or tap Try a sample.",
            (Button(label="Try a sample", value="/sample"),),
        )

    async def _violations(self, context: TurnContext, arguments: RecordingArguments) -> ToolResult:
        line = await resolve_line(self._repository, context.workspace_id, arguments.line)
        numbers = await self._analysis.rule_numbers(line.id)
        parts: list[str] = []
        for footage in await resolve_footage(self._repository, line, arguments.recording):
            found = await self._repository.violations(footage.id)
            name = f"{footage_name(footage)} ({duration_text(footage.duration_s)})"
            if not found:
                parts.append(f"{name}: no violations found (or not checked yet)")
                continue
            listed = "\n".join(
                f"{index}. {violation_line(v, numbers.get(v.rule_id))}"
                for index, v in enumerate(found, 1)
            )
            parts.append(f"{name}: {len(found)} violations\n{listed}")
        return ToolResult("\n\n".join(parts))

    async def _clip(self, context: TurnContext, arguments: ClipArguments) -> ToolResult:
        line = await resolve_line(self._repository, context.workspace_id, arguments.line)
        (footage,) = (await resolve_footage(self._repository, line, arguments.recording))[-1:]
        clip, buttons = await self._make_clip(context, footage, arguments.number)
        if not clip.media:
            return ToolResult(f"no clip can be sent here: {clip.caption}", buttons)
        return ToolResult(
            f"the clip is attached to your reply: {clip.caption}", buttons, clip.media
        )

    async def _ask(self, context: TurnContext, arguments: AskArguments) -> ToolResult:
        line = await resolve_line(self._repository, context.workspace_id, arguments.line)
        (footage,) = (await resolve_footage(self._repository, line, arguments.recording))[-1:]
        window = None
        if arguments.from_s is not None or arguments.to_s is not None:
            start = max(0.0, arguments.from_s or 0.0)
            end = min(footage.duration_s, arguments.to_s or footage.duration_s)
            if end <= start:
                raise ToolError(
                    f"{footage_name(footage)} is {video_length(footage.duration_s)} long; "
                    "that stretch is not in it"
                )
            window = TimeSpan(start=start, end=end)
        answer = await self._analysis.ask(context.workspace_id, footage, arguments.question, window)
        if answer is None:
            raise ToolError(f"{footage_name(footage)} has not been checked yet; check it first")
        buttons: tuple[Button, ...] = ()
        if answer.insight is not None:
            value = f"/insight {footage.id} {answer.insight.id}"
            buttons = (Button(label=f"▶ {answer.insight.label}", value=value),)
        return ToolResult(answer.text, buttons, reply=answer.text)

    async def _make_clip(
        self, context: TurnContext, footage: Footage, number: int
    ) -> tuple[Clip, tuple[Button, ...]]:
        found = await self._repository.violations(footage.id)
        if not 1 <= number <= len(found):
            raise ToolError(f"{footage_name(footage)} has {len(found)} violations")
        max_bytes = self._capabilities[context.chat.channel].max_media_bytes
        clip = await self._analysis.clip(footage, found[number - 1], number, len(found), max_bytes)
        return clip, clip_buttons(footage, number, len(found))
