import secrets
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

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
)
from rewind.core.analysis import (
    ANALYZE_JOB,
    Analysis,
    Clip,
    clip_buttons,
    duration_text,
    footage_name,
    violation_line,
)
from rewind.core.budget import BudgetExceededError
from rewind.core.conversation.agent import AgentReply
from rewind.core.conversation.conversation import Command
from rewind.core.conversation.store import ConversationStore
from rewind.core.conversation.tools import (
    ToolArguments,
    ToolError,
    ToolResult,
    ToolSpec,
    TurnContext,
)
from rewind.core.conversation.workspace_tools import resolve_line
from rewind.core.ports import JobPayload, JobRunner, Repository
from rewind.core.rules import RuleCompileError, RuleCompiler, normalize, zone_keys

CONFIRM_TOKEN = "analyze"
CONFIRM_TTL = timedelta(hours=1)
ALL = "all"


@dataclass(frozen=True, slots=True)
class AnalyzeSettings:
    # D-7: an estimate above this waits for a tap; the daily cap is the ledger's.
    confirm_above_usd: Decimal
    daily_cap_usd: Decimal


class AddRuleArguments(ToolArguments):
    text: str
    line: str | None = None


class RecordingArguments(ToolArguments):
    line: str | None = None
    # A recording's file name, or "all"; the newest recording when left out.
    recording: str | None = None


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


def add_rule(
    repository: Repository, compiler_for: Callable[[str], RuleCompiler]
) -> ToolSpec[AddRuleArguments]:
    async def run(context: TurnContext, arguments: AddRuleArguments) -> ToolResult:
        line = await resolve_line(repository, context.workspace_id, arguments.line)
        normalized = normalize(arguments.text)
        active = [rule for rule in await repository.rules(line.id) if rule.active]
        for index, rule in enumerate(active, 1):
            if rule.normalized_text == normalized:
                return ToolResult(f"line {line.name!r} already has this rule as rule {index}")
        try:
            compiled = await compiler_for(context.workspace_id).compile(arguments.text)
        except RuleCompileError as error:
            raise ToolError(f"I could not turn that into a rule: {error}") from error
        except BudgetExceededError as error:
            raise ToolError(f"the spending limit stops me: {error}") from error
        spec = compiled.spec
        rule = Rule(
            id=uuid.uuid4().hex, line_id=line.id, text=arguments.text.strip(),
            normalized_text=normalized, spec=spec, active=not spec.unsupported,
            created_at=context.now,
        )  # fmt: skip
        await repository.save_rule(rule)
        described = describe_rule(spec, line, compiled.zone_phrases)
        if not spec.unsupported:
            return ToolResult(f"rule {len(active) + 1} added to line {line.name!r}:\n{described}")
        dropped = "; ".join(repr(clause) for clause in spec.unsupported)
        return ToolResult(
            f"NOT saved yet. These parts cannot be checked: {dropped}.\n"
            f"The part that can be checked:\n{described}\n"
            "Ask the person whether to save the part that can be checked; the buttons do it.",
            buttons=(
                Button(label="Save that part", value=f"/keep_rule {rule.id}"),
                Button(label="Don't save", value=f"/drop_rule {rule.id}"),
            ),
        )

    return ToolSpec(
        "add_rule",
        "Add a rule to a line. Pass the person's sentence exactly as written.",
        AddRuleArguments,
        run,
    )


class Analyze:
    def __init__(
        self,
        repository: Repository,
        store: ConversationStore,
        jobs: JobRunner,
        analysis: Analysis,
        capabilities: Mapping[str, ChannelCapabilities],
        settings: AnalyzeSettings,
    ) -> None:
        self._repository = repository
        self._store = store
        self._jobs = jobs
        self._analysis = analysis
        self._capabilities = capabilities
        self._settings = settings

    def tool(self) -> ToolSpec[RecordingArguments]:
        return ToolSpec(
            "analyze",
            "Estimate the cost of checking recordings of a line against its rules, and start "
            "the check (or offer a confirm button when it costs more than the threshold).",
            RecordingArguments,
            self._run,
        )

    async def confirm(self, context: TurnContext, argument: str) -> AgentReply:
        payload = await self._store.take_token(CONFIRM_TOKEN, argument, context.now)
        if payload is None:
            return AgentReply("That button has expired or was used already. Ask me again.")
        await self._jobs.submit(ANALYZE_JOB, payload)
        return AgentReply("Started. I'll message you here when it's done.")

    async def cancel(self, context: TurnContext, argument: str) -> AgentReply:
        if argument:
            await self._store.take_token(CONFIRM_TOKEN, argument, context.now)
        return AgentReply("OK, I won't check it.")

    async def _run(self, context: TurnContext, arguments: RecordingArguments) -> ToolResult:
        line = await resolve_line(self._repository, context.workspace_id, arguments.line)
        rules = tuple(r for r in await self._repository.rules(line.id) if r.active)
        if not rules:
            raise ToolError(f"line {line.name!r} has no rules yet; add one first")
        footage = await resolve_footage(self._repository, line, arguments.recording)
        estimate = await self._analysis.estimate(context.workspace_id, line, footage, rules)
        start_of_day = context.now.replace(hour=0, minute=0, second=0, microsecond=0)
        spent = await self._repository.spent_usd(start_of_day, context.workspace_id)
        total = estimate.total_usd
        facts = [
            f"{footage_name(f.footage)}: {f.frames} frames to look at, {f.cached_frames} "
            f"already seen (free), about ${f.usd:.2f}"
            for f in estimate.footage
        ]
        if estimate.zones_to_find:
            facts.append(
                f"finding {', '.join(estimate.zones_to_find)} in the video once: "
                f"${estimate.zones_usd:.3f}"
            )
        facts.append(f"total about ${total:.2f}")
        facts.append(f"spent today ${spent:.2f} of the ${self._settings.daily_cap_usd:.2f} limit")
        payload: JobPayload = {
            "workspace_id": context.workspace_id,
            "line_id": line.id,
            "footage_ids": [f.id for f in footage],
            "rule_ids": [r.id for r in rules],
            "channel": context.chat.channel,
            "chat_id": context.chat.chat_id,
            "max_media_bytes": self._capabilities[context.chat.channel].max_media_bytes,
        }
        if spent + total > self._settings.daily_cap_usd:
            facts.append("NOT started: it would pass today's limit")
            return ToolResult("\n".join(facts))
        if total <= self._settings.confirm_above_usd:
            await self._jobs.submit(ANALYZE_JOB, payload)
            facts.append(
                f"started at once (under ${self._settings.confirm_above_usd:.2f}); "
                "results will arrive in this chat"
            )
            return ToolResult("\n".join(facts))
        token = secrets.token_urlsafe(12)
        await self._store.put_token(CONFIRM_TOKEN, token, dict(payload), context.now + CONFIRM_TTL)
        facts.append("NOT started yet: waiting for the person to tap the confirm button")
        return ToolResult(
            "\n".join(facts),
            buttons=(
                Button(label=f"Check it (${total:.2f})", value=f"/confirm {token}"),
                Button(label="Not now", value=f"/cancel {token}"),
            ),
        )


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

    async def clip_command(self, context: TurnContext, argument: str) -> AgentReply:
        footage_id, _, number = argument.partition(" ")
        footage = await self._repository.get_footage(footage_id)
        if footage is None or not number.isdigit():
            return AgentReply("I can't find that clip any more.")
        clip, buttons = await self._make_clip(context, footage, int(number))
        return AgentReply("" if clip.media else clip.caption, buttons, clip.media)

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
                f"{index}. {violation_line(v, numbers)}" for index, v in enumerate(found, 1)
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

    async def _make_clip(
        self, context: TurnContext, footage: Footage, number: int
    ) -> tuple[Clip, tuple[Button, ...]]:
        found = await self._repository.violations(footage.id)
        if not 1 <= number <= len(found):
            raise ToolError(f"{footage_name(footage)} has {len(found)} violations")
        max_bytes = self._capabilities[context.chat.channel].max_media_bytes
        clip = await self._analysis.clip(footage, found[number - 1], number, len(found), max_bytes)
        return clip, clip_buttons(footage, number, len(found))


def rule_commands(repository: Repository) -> dict[str, Command]:
    async def keep(context: TurnContext, argument: str) -> AgentReply:
        rule = await repository.get_rule(argument)
        if rule is None:
            return AgentReply("I can't find that rule any more.")
        await repository.save_rule(rule.model_copy(update={"active": True}))
        return AgentReply("Saved. I'll check the part I can.")

    async def drop(context: TurnContext, argument: str) -> AgentReply:
        return AgentReply("OK, I didn't save it.")

    return {"/keep_rule": keep, "/drop_rule": drop}
