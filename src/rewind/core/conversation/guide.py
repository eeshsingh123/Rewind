from rewind.contracts import Button
from rewind.core.conversation.agent import AgentReply
from rewind.core.conversation.analysis_tools import RULE_EXAMPLE
from rewind.core.conversation.conversation import Command
from rewind.core.conversation.tools import TurnContext
from rewind.core.conversation.workspace_tools import resolve_line
from rewind.core.ports import Repository

# The chat app's command menu: (command, what it does).
MENU = (
    ("start", "Start over and try the sample"),
    ("rules", "See the rules of your line"),
    ("clips", "Clips from the last check"),
    ("help", "What I can do"),
)
TRY_SAMPLE = Button(label="Try a sample", value="/sample")
ADD_RULE = Button(label="Add a rule", value="/add_rule")


# Fixed answers that need no model call, so they are instant and free. Each ends with what to
# do next.
def guide_commands(repository: Repository, sample: bool) -> dict[str, Command]:
    sample_buttons = (TRY_SAMPLE,) if sample else ()

    async def start(context: TurnContext, argument: str) -> AgentReply:
        text = (
            "Hi, I'm *Rewind*. I watch footage of a packing or production line and tell you "
            "which items broke your rules, with a short clip of each one.\n\n"
        )
        if sample:
            text += "To try it, tap *Try a sample*. "
        text += "To use your own line, send me a video and one rule in plain words, in any order."
        return AgentReply(text, sample_buttons)

    async def help_(context: TurnContext, argument: str) -> AgentReply:
        text = (
            f"Send me a video of your line and a rule in plain words, like {RULE_EXAMPLE}. "
            "I'll check every item and send you a clip of each one that broke the rule.\n\n"
            "Rules also work on videos you sent before.\n\n"
            "/rules · your rules\n/clips · clips from the last check"
        )
        return AgentReply(text, sample_buttons)

    async def rules(context: TurnContext, argument: str) -> AgentReply:
        line = await resolve_line(repository, context.workspace_id, None)
        active = [rule for rule in await repository.rules(line.id) if rule.active]
        if not active:
            return AgentReply(
                f"*{line.name}* has no rules yet. Tell me one in plain words, for example: "
                f"{RULE_EXAMPLE}."
            )
        listed = "\n".join(f"{index}. {rule.text}" for index, rule in enumerate(active, 1))
        return AgentReply(f"Rules of *{line.name}*:\n{listed}", (ADD_RULE,))

    async def add_rule(context: TurnContext, argument: str) -> AgentReply:
        return AgentReply(f"Type the rule in one sentence, for example: {RULE_EXAMPLE}.")

    async def send_video(context: TurnContext, argument: str) -> AgentReply:
        return AgentReply(
            "Send me a video of your line. If it's too big for this chat, I'll give you an "
            "upload link."
        )

    return {
        "/start": start,
        "/help": help_,
        "/rules": rules,
        "/add_rule": add_rule,
        "/send_video": send_video,
    }
