import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from rewind.contracts import Button, ReplyMedia
from rewind.core.ports import ChannelError

logger = logging.getLogger(__name__)

# A count ("4 of 6") changes often; the chat is edited at most this often for it.
COUNT_EVERY_S = 2.0

type StepState = Literal["waiting", "running", "done", "skipped"]
_MARKS: dict[StepState, str] = {"waiting": "○", "running": "⏳", "done": "✓", "skipped": "·"}


class Notifier(Protocol):
    async def notify(
        self,
        channel: str,
        chat_id: str,
        text: str,
        media: tuple[ReplyMedia, ...] = (),
        buttons: tuple[Button, ...] = (),
    ) -> None: ...

    async def progress(
        self, channel: str, chat_id: str, key: str, text: str, buttons: tuple[Button, ...] = ()
    ) -> None: ...


@dataclass(slots=True)
class Step:
    key: str
    # "Finding boxes", shown before the step starts and, with "…", while it runs.
    doing: str
    # "Found 6 boxes"; set when the step ends.
    done: str = ""
    state: StepState = "waiting"
    count: tuple[int, int] | None = None

    def line(self) -> str:
        mark = _MARKS[self.state]
        if self.state in ("done", "skipped"):
            return f"{mark} {self.done or self.doing}"
        if self.state == "waiting":
            return f"{mark} {self.doing}"
        counted = (
            f" ({self.count[0]} of {self.count[1]})" if self.count and self.count[1] > 1 else ""
        )
        return f"{mark} {self.doing}…{counted}"


# One chat message, edited in place, that ticks off the real stages of a check as they end.
# Only a cue: if the chat refuses an edit, the check goes on.
class Checklist:
    def __init__(
        self,
        notifier: Notifier,
        channel: str,
        chat_id: str,
        key: str,
        header: str,
        steps: Sequence[Step],
    ) -> None:
        self._notifier = notifier
        self._channel = channel
        self._chat_id = chat_id
        self._key = key
        self.header = header
        self._steps = {step.key: step for step in steps}
        self._shown = ""
        self._shown_at = 0.0

    def text(self) -> str:
        lines = "\n".join(step.line() for step in self._steps.values())
        return f"{self.header}\n\n{lines}"

    async def show(self) -> None:
        text = self.text()
        if text == self._shown:
            return
        self._shown, self._shown_at = text, time.monotonic()
        try:
            await self._notifier.progress(self._channel, self._chat_id, self._key, text)
        except ChannelError:
            logger.warning("could not update progress in chat %s", self._chat_id, exc_info=True)

    async def start(self, key: str) -> None:
        step = self._steps[key]
        step.state, step.count = "running", None
        await self.show()

    async def count(self, key: str, done: int, total: int) -> None:
        step = self._steps[key]
        step.state, step.count = "running", (done, total)
        if time.monotonic() - self._shown_at >= COUNT_EVERY_S:
            await self.show()

    async def finish(self, key: str, done: str) -> None:
        step = self._steps[key]
        step.state, step.done, step.count = "done", done, None
        await self.show()

    async def skip(self, key: str, why: str) -> None:
        step = self._steps[key]
        step.state, step.done = "skipped", why
        await self.show()

    def has(self, key: str) -> bool:
        return key in self._steps


# "about 3 min", from an estimate in seconds; never a false precision.
def about_time(seconds: float) -> str:
    if seconds < 60:
        return "under a minute"
    return f"about {math.ceil(seconds / 60)} min"
