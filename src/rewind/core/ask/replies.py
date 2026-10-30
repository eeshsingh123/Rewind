import re
from collections.abc import Sequence
from decimal import Decimal
from urllib.parse import urlsplit

from rewind.contracts import Button, ChannelCapabilities, TimeSpan
from rewind.core.analysis import clock_time
from rewind.core.ask.session import Exchange, Session, Step

MENU = (
    ("start", "How this works"),
    ("help", "What I can do"),
)
START = (
    "Hi, I'm *Rewind*. Send me a photo or a video and ask anything about it, by text or by "
    "voice note in any language. I'll check and answer with proof."
)
HELP = (
    "Send me a photo or a video. I'll tell you what stands out and suggest questions.\n\n"
    "Then ask anything, by text or voice note, or tap a question. Each answer comes with its "
    "proof. Your next message is always about the latest photo or video."
)
NO_FOOTAGE = "Send me a photo or a video first, then ask anything about it."
NOT_ALLOWED = "Sorry, this assistant is private."
NOT_FOOTAGE = "I can only read photos and videos. Please send one of those."
NOT_FETCHED = "I couldn't download that. Please send it again."
NOT_HEARD = "I couldn't make out any words in that voice note. Please try again."
BUSY = "I'm still working on the last question. Ask again once it's answered."
FETCHING = "Getting your file"
WORKING = "On it"
_PROOF_TOO_BIG = "The proof is too big to send here."
_PROOF_ON_TIMELINE = "The proof is too big to send here, so it's on the timeline."
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})
_QUESTION_SHOWN_CHARS = 200
_TAP = re.compile(r"q:([\w-]+):(\d+)")


def too_big(limit_bytes: int, base_url: str) -> str:
    said = f"That file is too big for this chat (the limit is {limit_bytes // 1_000_000} MB)."
    if _is_local(base_url):
        return f"{said} Please send a shorter clip."
    return f"{said} Upload it on the web page instead: {base_url.rstrip('/')}"


def too_long(max_seconds: int) -> str:
    return f"That voice note is too long; the limit is {max_seconds // 60} minutes."


def timeline_link(base_url: str, session_id: str) -> str | None:
    return None if _is_local(base_url) else f"{base_url.rstrip('/')}/s/{session_id}"


def first_look_text(session: Session, capabilities: ChannelCapabilities) -> str:
    look = session.first_look
    head = [look.summary]
    if look.moments:
        head += ["", "*What stands out*"]
    moments = [f"• {_moment(session.photo, m.span, m.what)}" for m in look.moments]
    tail = ["", "Ask anything, or tap a question."] if look.questions else ["", "Ask anything."]
    return _fit(head, moments, tail, capabilities)


# A tap names its session and the question's place: a value short enough for every channel, and
# one that still means the same question after newer footage has come in.
def question_buttons(session: Session, capabilities: ChannelCapabilities) -> tuple[Button, ...]:
    questions = session.first_look.questions[: capabilities.max_buttons]
    return tuple(
        Button(label=question, value=f"q:{session.id}:{index}")
        for index, question in enumerate(questions)
    )


def tapped(text: str) -> tuple[str, int] | None:
    match = _TAP.fullmatch(text)
    return None if match is None else (match.group(1), int(match.group(2)))


def progress_text(question: str | None, steps: Sequence[Step]) -> str:
    lines = [f"✓ {step.text}" for step in steps if step.done]
    if steps and not steps[-1].done:
        lines.append(f"⏳ {steps[-1].text}…")
    if question:
        lines.insert(0, f"*{question[:_QUESTION_SHOWN_CHARS]}*")
    return "\n".join(lines)


def answer_text(
    exchange: Exchange,
    timeline: str | None,
    proof_left_out: bool,
    capabilities: ChannelCapabilities,
) -> str:
    tail = ["", _cost_line(exchange.cost_usd)]
    if proof_left_out:
        tail.append(_PROOF_ON_TIMELINE if timeline else _PROOF_TOO_BIG)
    if timeline:
        tail.append(f"Open the full timeline: {timeline}")
    return _fit([], exchange.answer.splitlines(), tail, capabilities)


def _is_local(base_url: str) -> bool:
    return (urlsplit(base_url).hostname or "") in _LOCAL_HOSTS


def _moment(photo: bool, span: TimeSpan, what: str) -> str:
    if photo:
        return what
    start, end = clock_time(span.start), clock_time(span.end)
    return f"{start}: {what}" if start == end else f"{start} to {end}: {what}"


# Same words as the web page's cost line.
def _cost_line(cost_usd: Decimal) -> str:
    if cost_usd == 0:
        return "No cost: already answered"
    return "cost under $0.001" if cost_usd < Decimal("0.001") else f"cost ${cost_usd:.3f}"


# As many of the middle lines as fit, so what comes last is never the part that is cut.
def _fit(
    head: Sequence[str],
    lines: Sequence[str],
    tail: Sequence[str],
    capabilities: ChannelCapabilities,
) -> str:
    kept = list(lines)
    while kept:
        text = "\n".join([*head, *kept, *tail])
        if len(text) <= capabilities.max_text_chars:
            return text
        kept.pop()
    return "\n".join([*head, *tail])[: capabilities.max_text_chars]
