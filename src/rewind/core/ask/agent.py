import hashlib
import json
import logging
from dataclasses import dataclass
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rewind.contracts import Charge, CostEstimate, TimeSpan
from rewind.core.ask.areas import Area
from rewind.core.ask.evidence import Claim, Notebook
from rewind.core.ask.proofs import Proofs
from rewind.core.ask.session import Exchange, Finding, Marked, OnStep, Session, Sessions, Step
from rewind.core.ask.tools import MARKED_THING, AskTool, Bench, Segmenters, Started, ToolError
from rewind.core.ask.words import clock_time
from rewind.core.ports import (
    AssistantMessage,
    ChatMessage,
    CompletionRequest,
    Effort,
    LanguageModel,
    LanguageModelError,
    SegmenterError,
    SystemMessage,
    Tool,
    ToolResultMessage,
    UserMessage,
    VideoToolsError,
)

logger = logging.getLogger(__name__)

ANSWER = "answer"
MARKED_PART = "the part you marked"
_OUTPUT_UNITS = frozenset({"output_token", "reasoning_token"})
_ANSWER_VERSION = "2"
NOTHING_BACKED = "I couldn't back an answer to that with what I found."
NUDGE = "Give your answer by calling answer, citing the evidence ids behind each sentence."

INSTRUCTIONS = """\
You answer questions about one {media} for the person in charge of the place it shows. You never \
see the {media} yourself: you use tools, and each tool result is evidence with an id.

{watch}- find: finds every item of one kind of thing and names each {named}. Use it when a \
question is about separate items. It costs the most; call it once per thing.
- look: a close look at each item of a found thing, or at a whole {frame} or an area, with one \
question for each ("Does it have a label?"). Use it to judge items one by one.
- measure: {measure}
{follow}- area: marks one named part of the {frame} ("the left half"){when}, so find, look and \
measure can work on that part alone by naming it. Use it when a question is about one part of \
the {frame}. If the area cannot be made out, say so plainly and do not guess where it is.
- answer: your reply, as claims.

The first look below is a quick pass over the whole {media}: use it to know where to look, and \
confirm claims about single items with find and look before stating them. Plan a few steps, \
then answer. Do not call a tool again with the same arguments.

Answer with answer. Each claim is one short plain sentence for a manager, not an engineer: \
{places}, items named as tools named them, no ids, no tool names. Each claim cites the \
evidence ids that back it, and states numbers and times only as that evidence writes them. Lead \
with the direct answer. When the evidence cannot answer something, say so plainly in a claim \
citing what you looked at. Use as few claims as answer the question. Write every claim in the \
language of the question, with numbers always in digits, never spelled out."""
VIDEO_WORDS = {
    "media": "video",
    "frame": "frame",
    "watch": "- watch: Muse watches the video or a stretch of it. Good for what happens, when, "
    "and why. It can miss single items or miscount them.\n",
    "named": 'by when it is seen best ("the item at 00:14")',
    "measure": "code counts a thing's items and times them: pace, stops in the flow, time in view.",
    "when": " on the frame at a time you give",
    "places": "times as mm:ss",
}
PHOTO_WORDS = {
    "media": "photo",
    "frame": "photo",
    "watch": "",
    "named": 'by where it is ("the 2nd item from the left")',
    "measure": "code counts a thing's items.",
    "when": "",
    "places": "say where things are, never a time",
}


FOLLOW_WORDS = (
    "- follow: tracks what the person marked through the whole video: when it is in view and "
    "where it goes.\n"
)


class _ClaimArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    text: str = Field(description="One short plain sentence.")
    evidence: list[str] = Field(description="Ids of the evidence that backs it: ['e3', 'e7'].")


class _AnswerArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    claims: list[_ClaimArguments] = Field(min_length=1)


ANSWER_TOOL = Tool(
    ANSWER,
    "Your reply to the question, as claims that each cite their evidence.",
    _AnswerArguments.model_json_schema(),
)


@dataclass(frozen=True, slots=True)
class AskLimits:
    # A question can never loop or spend without bound. The budget is checked before each step.
    budget_usd: Decimal = Decimal("0.50")
    max_tool_calls: int = 12
    max_output_tokens: int = 12_000
    effort: Effort = "low"


class AskAgent:
    def __init__(
        self,
        sessions: Sessions,
        tools: tuple[AskTool, ...],
        proofs: Proofs,
        limits: AskLimits,
    ) -> None:
        self._sessions = sessions
        self._tools = {tool.name: tool for tool in tools}
        self._proofs = proofs
        self._limits = limits

    # True when this question would be answered from memory, at no cost.
    async def knows(self, session: Session, question: str, model: LanguageModel) -> bool:
        digest = _digest(session, question, None, model)
        return await self._sessions.remembered(digest) is not None

    async def ask(
        self,
        session: Session,
        question: str,
        model: LanguageModel,
        segmenters: Segmenters,
        on_step: OnStep,
        marked: Marked | None = None,
    ) -> Exchange:
        digest = _digest(session, question, marked, model)
        exchange = await self._sessions.remembered(digest)
        if exchange is not None:
            for step in exchange.steps:
                await on_step(Step(text=step, done=True))
            exchange = exchange.model_copy(update={"question": question, "cost_usd": Decimal(0)})
        else:
            exchange = await self._work_out(session, question, model, segmenters, on_step, marked)
            if exchange.findings:
                await self._sessions.remember(digest, exchange)
        await self._sessions.save(
            session.model_copy(update={"exchanges": (*session.exchanges, exchange)})
        )
        return exchange

    async def _work_out(
        self,
        session: Session,
        question: str,
        model: LanguageModel,
        segmenters: Segmenters,
        on_step: OnStep,
        marked: Marked | None,
    ) -> Exchange:
        notebook = Notebook()
        bench = Bench(session, model, segmenters, notebook, marked=marked)
        if marked is not None:
            bench.areas[MARKED_PART] = Area(MARKED_PART, marked.box, marked.t)
        follow = FOLLOW_WORDS if marked is not None and not session.photo else ""
        words = PHOTO_WORDS if session.photo else VIDEO_WORDS
        instructions = INSTRUCTIONS.format(**words, follow=follow)
        state = _state(session, bench)
        messages: list[ChatMessage] = [SystemMessage(instructions), SystemMessage(state)]
        for exchange in session.exchanges:
            messages += [UserMessage(exchange.question), AssistantMessage(exchange.answer)]
        messages.append(UserMessage(question))
        steps: list[str] = []

        async def noted(step: Step) -> None:
            if step.done:
                steps.append(step.text)
            await on_step(step)

        run = _Run(self._limits, bench)
        claims = await self._loop(bench, messages, run, noted)
        if claims:
            await on_step(Step(text="Drawing the proof", done=False))
        findings: list[Finding] = []
        for claim in claims:
            shown = claim.shown
            proof = await self._proofs.render(session, shown) if shown else None
            span = None if session.photo or shown is None else shown.clip
            findings.append(Finding(text=claim.text, kind=claim.kind, proof_key=proof, span=span))
        return Exchange(
            question=question,
            answer="\n".join(f.text for f in findings) or NOTHING_BACKED,
            findings=tuple(findings),
            steps=tuple(steps),
            cost_usd=run.spent,
            marked=marked,
        )

    async def _loop(
        self, bench: Bench, messages: list[ChatMessage], run: "_Run", on_step: OnStep
    ) -> tuple[Claim, ...]:
        nudged = last_chance = False
        kept: tuple[Claim, ...] = ()
        # Once the steps or the budget run out, the agent gets one last call, to answer.
        while run.tokens_left > 0 and not last_chance:
            last_chance = not run.can_act()
            completion = await bench.model.complete(
                CompletionRequest(
                    messages=tuple(messages),
                    max_output_tokens=run.tokens_left,
                    tools=self._offered(bench, run),
                    effort=self._limits.effort,
                )
            )
            run.think(completion.charges)
            if not completion.tool_calls:
                if nudged:
                    return kept
                nudged = True
                messages += [AssistantMessage(completion.text), UserMessage(NUDGE)]
                continue
            messages.append(AssistantMessage(completion.text, completion.tool_calls))
            for call in completion.tool_calls:
                if call.name == ANSWER:
                    claims, problems = _claims(bench.notebook, call.arguments)
                    if not problems or run.answered:
                        return claims
                    run.answered, kept = True, claims
                    result = "Some claims were not backed:\n" + "\n".join(problems)
                    result += "\nCall answer again with every claim fixed or left out."
                else:
                    result = await self._call(bench, call.name, call.arguments, run, on_step)
                messages.append(ToolResultMessage(call.id, result))
        return kept

    def _offered(self, bench: Bench, run: "_Run") -> tuple[Tool, ...]:
        if not run.can_act():
            return (ANSWER_TOOL,)
        usable = [tool for tool in self._tools.values() if _usable(tool, bench)]
        return (*(tool.as_tool() for tool in usable), ANSWER_TOOL)

    async def _call(
        self, bench: Bench, name: str, arguments: str, run: "_Run", on_step: OnStep
    ) -> str:
        tool = self._tools.get(name)
        if tool is None or not _usable(tool, bench):
            return f"error: there is no tool named {name!r}"
        if not run.can_act():
            return "error: no more steps for this question; answer with what you have"
        started: Started | str = tool.start(arguments)
        if isinstance(started, str):
            return f"error: {started}"
        run.tool_calls += 1
        await on_step(Step(text=started.doing, done=False))
        try:
            outcome = await started.run(bench)
        except ToolError as error:
            return f"error: {error}"
        except (SegmenterError, LanguageModelError, VideoToolsError) as error:
            logger.exception("%s failed", name)
            await on_step(Step(text=f"{started.doing}: that did not work", done=True))
            return f"error: {name} failed ({error}); try another way or answer with what you have"
        await on_step(Step(text=outcome.done, done=True))
        return outcome.text


class _Run:
    def __init__(self, limits: AskLimits, bench: Bench) -> None:
        self._limits = limits
        self._bench = bench
        self.tool_calls = 0
        # The agent's own output only: tools have their own limits per call.
        self.tokens_left = limits.max_output_tokens
        self.answered = False

    @property
    def spent(self) -> Decimal:
        return CostEstimate(charges=tuple(self._bench.charges)).total_usd

    def think(self, charges: tuple[Charge, ...]) -> None:
        self._bench.charges.extend(charges)
        self.tokens_left -= sum(c.quantity for c in charges if c.unit in _OUTPUT_UNITS)

    def can_act(self) -> bool:
        return (
            self.tool_calls < self._limits.max_tool_calls and self.spent < self._limits.budget_usd
        )


def _usable(tool: AskTool, bench: Bench) -> bool:
    if tool.video_only and bench.session.photo:
        return False
    return bench.marked is not None or not tool.needs_mark


# The same question on the same footage is never paid for twice. A suggested question stands
# alone, so it matches in any order; any other may lean on what was asked before it.
def _digest(session: Session, question: str, marked: Marked | None, model: LanguageModel) -> str:
    before = () if question in session.first_look.questions else session.exchanges
    inputs = [
        session.footage.content_hash,
        [(e.question, e.marked.model_dump() if e.marked else None) for e in before],
        " ".join(question.lower().split()),
        marked.model_dump() if marked else None,
        model.model_version,
        _ANSWER_VERSION,
    ]
    return hashlib.sha256(json.dumps(inputs).encode()).hexdigest()[:24]


def _claims(notebook: Notebook, raw: str) -> tuple[tuple[Claim, ...], list[str]]:
    try:
        arguments = _AnswerArguments.model_validate_json(raw or "{}")
    except ValidationError as error:
        return (), [f"invalid arguments for answer: {error}"]
    kept: list[Claim] = []
    problems: list[str] = []
    for proposed in arguments.claims:
        checked = notebook.check(proposed.text, proposed.evidence)
        if isinstance(checked, str):
            logger.info("dropped a claim (%s): %s", checked, proposed.text)
            problems.append(f'"{proposed.text}": {checked}')
        elif all(claim.text != checked.text for claim in kept):
            kept.append(checked)
    return tuple(kept), problems


# What the agent knows before any tool: the video's length, Muse's first look as evidence, a box
# the person drew, and the questions already asked. On a photo the first look's proof is the photo.
def _state(session: Session, bench: Bench) -> str:
    notebook = bench.notebook
    look = session.first_look
    footage = session.footage
    if session.photo:
        whole = TimeSpan(start=0, end=0)
        return "\n".join([
            "This is one photo: nothing in it has a time.",
            "First look (Muse looked at the whole photo once):",
            notebook.add("seen", f"In the whole photo: {look.summary}", whole).line,
            *(notebook.add("seen", m.what, whole).line for m in look.moments),
            *_marked_lines(bench),
        ])  # fmt: skip
    lines = [
        f"The video is {clock_time(footage.duration_s)} long.",
        "First look (Muse watched the whole video once):",
        notebook.add("seen", f"In the whole video: {look.summary}").line,
    ]
    for moment in look.moments:
        clip = moment.span
        text = f"{clock_time(clip.start)} to {clock_time(clip.end)}: {moment.what}"
        lines.append(notebook.add("seen", text, clip).line)
    return "\n".join([*lines, *_marked_lines(bench)])


def _marked_lines(bench: Bench) -> list[str]:
    area = bench.areas.get(MARKED_PART)
    if area is None:
        return []
    where = "the photo" if bench.session.photo else f"the frame at {clock_time(area.t)}"
    instant = TimeSpan(start=area.t, end=area.t)
    text = f"The person marked a part of {where}."
    lines = [
        f'The person drew a box on {where}: it is the area "{MARKED_PART}". To ask about that '
        "part, pass it as area to find, look and measure.",
        bench.notebook.add("seen", text, instant, (area.mark,)).line,
    ]
    if not bench.session.photo:
        lines.append(
            "To ask about the thing in the box, use follow; look can then judge it as the thing "
            f'"{MARKED_THING}".'
        )
    return lines
