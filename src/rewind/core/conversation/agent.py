from dataclasses import dataclass

from rewind.contracts import Button, ReplyMedia
from rewind.core.conversation.store import HistoryEntry
from rewind.core.conversation.tools import Job, Toolbox, TurnContext
from rewind.core.ports import (
    AssistantMessage,
    ChatMessage,
    CompletionRequest,
    Effort,
    LanguageModel,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)

_OUTPUT_UNITS = frozenset({"output_token", "reasoning_token"})
STOPPED = "I stopped there to keep this turn small. Ask me again to continue."


@dataclass(frozen=True, slots=True)
class TurnLimits:
    # D-7: a turn can never loop or spend without bound.
    max_tool_calls: int = 6
    max_output_tokens: int = 4000
    effort: Effort = "low"


@dataclass(frozen=True, slots=True)
class AgentReply:
    text: str
    buttons: tuple[Button, ...] = ()
    media: tuple[ReplyMedia, ...] = ()
    jobs: tuple[Job, ...] = ()


class Agent:
    def __init__(self, instructions: str, toolbox: Toolbox, limits: TurnLimits) -> None:
        self._instructions = instructions
        self._toolbox = toolbox
        self._limits = limits

    async def run(
        self,
        model: LanguageModel,
        context: TurnContext,
        state: str,
        history: tuple[HistoryEntry, ...],
        text: str,
    ) -> AgentReply:
        # The fixed instructions come first so the provider can cache them across turns.
        messages: list[ChatMessage] = [SystemMessage(self._instructions), SystemMessage(state)]
        messages += [_message(entry) for entry in history]
        messages.append(UserMessage(text))
        buttons: list[Button] = []
        media: list[ReplyMedia] = []
        jobs: list[Job] = []
        tool_calls_left = self._limits.max_tool_calls
        tokens_left = self._limits.max_output_tokens
        tools = self._toolbox.as_tools()
        while tokens_left > 0:
            completion = await model.complete(
                CompletionRequest(
                    messages=tuple(messages),
                    max_output_tokens=tokens_left,
                    tools=tools if tool_calls_left > 0 else (),
                    effort=self._limits.effort,
                )
            )
            tokens_left -= sum(c.quantity for c in completion.charges if c.unit in _OUTPUT_UNITS)
            if not completion.tool_calls:
                text = completion.text.strip() or STOPPED
                return AgentReply(text, tuple(buttons), tuple(media), tuple(jobs))
            messages.append(AssistantMessage(completion.text, completion.tool_calls))
            for call in completion.tool_calls:
                if tool_calls_left <= 0:
                    result_text = (
                        "error: no more tool calls in this turn; answer with what you have"
                    )
                else:
                    tool_calls_left -= 1
                    result = await self._toolbox.call(context, call.name, call.arguments)
                    result_text = result.text
                    buttons += result.buttons
                    media += result.media
                    jobs += result.jobs
                messages.append(ToolResultMessage(call.id, result_text))
        return AgentReply(STOPPED, tuple(buttons), tuple(media), tuple(jobs))


def _message(entry: HistoryEntry) -> ChatMessage:
    if entry.role == "user":
        return UserMessage(entry.text)
    return AssistantMessage(entry.text)
