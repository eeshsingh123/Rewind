import logging
from dataclasses import dataclass
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rewind.contracts import Charge, CostEstimate
from rewind.core.analysis import clock_time
from rewind.core.ask.clips import ProofClips
from rewind.core.ask.evidence import Claim, Kind, Notebook
from rewind.core.ask.session import Exchange, OnStep, Session, Sessions, Step
from rewind.core.ask.tools import AskTool, Bench, Started, ToolError
from rewind.core.budget import BudgetExceededError
from rewind.core.ports import (
    AssistantMessage,
    ChatMessage,
    CompletionRequest,
    Effort,
    LanguageModel,
    LanguageModelError,
    Segmenter,
    SegmenterError,
    SystemMessage,
    Tool,
    ToolResultMessage,
    UserMessage,
    VideoToolsError,
)

logger = logging.getLogger(__name__)

ANSWER = "answer"
_OUTPUT_UNITS = frozenset({"output_token", "reasoning_token"})
NOTHING_BACKED = "I couldn't back an answer to that with what I found in the video."
NUDGE = "Give your answer by calling answer, citing the evidence ids behind each sentence."

INSTRUCTIONS = """\
You answer questions about one video for the person in charge of the place it shows. You never \
see the video yourself: you use tools, and each tool result is evidence with an id.

- watch: Muse watches the video or a stretch of it. Good for what happens, when, and why. It \
can miss single items or miscount them.
- find: tracks every item of one kind of thing and names each by when it is seen best ("the jar \
at 00:14"). Use it when a question is about separate items. It costs the most; call it once per \
thing.
- look: a close look at each item of a found thing, or at whole frames, with one question for \
each ("Does it have a label?"). Use it to judge items one by one.
- measure: code counts a thing's items and times them: pace, stops in the flow, time in view.
- answer: your reply, as claims.

The first look below is a quick pass over the whole video: use it to know where to look, and \
confirm claims about single items with find and look before stating them. Plan a few steps, \
then answer. Do not call a tool again with the same arguments.

Answer with answer. Each claim is one short plain sentence for a manager, not an engineer: \
times as mm:ss, items named as tools named them, no ids, no tool names. Each claim cites the \
evidence ids that back it, and states numbers and times only as that evidence writes them. Lead \
with the direct answer. When the evidence cannot answer something, say so plainly in a claim \
citing what you looked at. Use as few claims as answer the question."""


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
class Finding:
    text: str
    kind: Kind
    # The proof clip's blob key; None when the evidence has no moment to show.
    clip_key: str | None


@dataclass(frozen=True, slots=True)
class Answer:
    findings: tuple[Finding, ...]
    cost_usd: Decimal

    @property
    def text(self) -> str:
        return "\n".join(f.text for f in self.findings) or NOTHING_BACKED


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
        clips: ProofClips,
        limits: AskLimits,
    ) -> None:
        self._sessions = sessions
        self._tools = {tool.name: tool for tool in tools}
        self._clips = clips
        self._limits = limits

    async def ask(
        self,
        session: Session,
        question: str,
        model: LanguageModel,
        segmenter: Segmenter,
        on_step: OnStep,
    ) -> Answer:
        notebook = Notebook()
        bench = Bench(session, model, segmenter, notebook)
        state = _state(session, notebook)
        messages: list[ChatMessage] = [SystemMessage(INSTRUCTIONS), SystemMessage(state)]
        for exchange in session.exchanges:
            messages += [UserMessage(exchange.question), AssistantMessage(exchange.answer)]
        messages.append(UserMessage(question))
        run = _Run(self._limits, bench)
        claims: tuple[Claim, ...] = ()
        try:
            claims = await self._loop(bench, messages, run, on_step)
        except BudgetExceededError:
            logger.warning("session %s hit a spending cap", session.id)
        findings: list[Finding] = []
        for claim in claims:
            shown = claim.shown
            clip = await self._clips.render(session.footage, shown) if shown else None
            findings.append(Finding(claim.text, claim.kind, clip))
        answer = Answer(tuple(findings), run.spent)
        exchanges = (*session.exchanges, Exchange(question=question, answer=answer.text))
        await self._sessions.save(session.model_copy(update={"exchanges": exchanges}))
        return answer

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
                    tools=self._offered(run),
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

    def _offered(self, run: "_Run") -> tuple[Tool, ...]:
        if not run.can_act():
            return (ANSWER_TOOL,)
        return (*(tool.as_tool() for tool in self._tools.values()), ANSWER_TOOL)

    async def _call(
        self, bench: Bench, name: str, arguments: str, run: "_Run", on_step: OnStep
    ) -> str:
        tool = self._tools.get(name)
        if tool is None:
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


# What the agent knows before any tool: the video's length, Muse's first look as evidence,
# and the questions already asked.
def _state(session: Session, notebook: Notebook) -> str:
    look = session.first_look
    footage = session.footage
    lines = [
        f"The video is {clock_time(footage.duration_s)} long.",
        "First look (Muse watched the whole video once):",
        notebook.add("seen", f"In the whole video: {look.summary}").line,
    ]
    for moment in look.moments:
        clip = moment.span
        text = f"{clock_time(clip.start)} to {clock_time(clip.end)}: {moment.what}"
        lines.append(notebook.add("seen", text, clip).line)
    return "\n".join(lines)
