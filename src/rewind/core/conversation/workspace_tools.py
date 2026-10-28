import secrets
import uuid
from datetime import timedelta

from rewind.contracts import Line
from rewind.core.conversation.store import ConversationStore
from rewind.core.conversation.tools import (
    ToolArguments,
    ToolError,
    ToolResult,
    ToolSpec,
    TurnContext,
)
from rewind.core.ports import Repository

UPLOAD_TOKEN = "upload"


class LineArguments(ToolArguments):
    line: str | None = None


class CreateLineArguments(ToolArguments):
    name: str


# The named line, or the newest one when no name is given.
async def resolve_line(repository: Repository, workspace_id: str, name: str | None) -> Line:
    lines = sorted(await repository.lines(workspace_id), key=lambda line: line.created_at)
    if not lines:
        raise ToolError("this workspace has no lines yet; create one first")
    if name is None:
        return lines[-1]
    wanted = name.strip().casefold()
    for line in lines:
        if line.name.casefold() == wanted:
            return line
    known = ", ".join(line.name for line in lines)
    raise ToolError(f"no line named {name!r}; the lines are: {known}")


def create_line(repository: Repository) -> ToolSpec[CreateLineArguments]:
    async def run(context: TurnContext, arguments: CreateLineArguments) -> ToolResult:
        name = arguments.name.strip()
        if not name:
            raise ToolError("a line needs a name")
        existing = {line.name.casefold() for line in await repository.lines(context.workspace_id)}
        if name.casefold() in existing:
            raise ToolError(f"a line named {name!r} already exists")
        line = Line(
            id=uuid.uuid4().hex, workspace_id=context.workspace_id, name=name,
            created_at=context.now,
        )  # fmt: skip
        await repository.save_line(line)
        return ToolResult(f"created line {name!r}; it has no rules and no footage yet")

    return ToolSpec(
        "create_line", "Create a new line (one camera's view of a production line).",
        CreateLineArguments, run,
    )  # fmt: skip


def list_rules(repository: Repository) -> ToolSpec[LineArguments]:
    async def run(context: TurnContext, arguments: LineArguments) -> ToolResult:
        line = await resolve_line(repository, context.workspace_id, arguments.line)
        rules = [rule for rule in await repository.rules(line.id) if rule.active]
        if not rules:
            return ToolResult(f"line {line.name!r} has no rules")
        listed = "\n".join(f"rule {index}: {rule.text}" for index, rule in enumerate(rules, 1))
        return ToolResult(f"rules of line {line.name!r}:\n{listed}")

    return ToolSpec("list_rules", "List the active rules of a line.", LineArguments, run)


def upload_link(
    repository: Repository, store: ConversationStore, base_url: str, ttl: timedelta
) -> ToolSpec[LineArguments]:
    async def run(context: TurnContext, arguments: LineArguments) -> ToolResult:
        line = await resolve_line(repository, context.workspace_id, arguments.line)
        url = await make_upload_link(store, base_url, ttl, context, line)
        hours = int(ttl.total_seconds() // 3600)
        return ToolResult(
            f"upload link for line {line.name!r}: {url} (one file, valid for {hours} hours)"
        )

    return ToolSpec(
        "upload_link",
        "Make a link where a long recording can be uploaded straight to a line.",
        LineArguments,
        run,
    )


async def make_upload_link(
    store: ConversationStore, base_url: str, ttl: timedelta, context: TurnContext, line: Line
) -> str:
    token = secrets.token_urlsafe(16)
    payload: dict[str, str] = {
        "workspace_id": context.workspace_id,
        "line_id": line.id,
        "channel": context.chat.channel,
        "chat_id": context.chat.chat_id,
    }
    await store.put_token(UPLOAD_TOKEN, token, dict(payload), context.now + ttl)
    return f"{base_url.rstrip('/')}/u/{token}"
