from dataclasses import dataclass

from rewind.core.conversation.store import ChatRef


@dataclass(frozen=True, slots=True)
class Reading:
    footage_id: str
    line_id: str
    name: str


@dataclass(frozen=True, slots=True)
class WaitingCheck:
    workspace_id: str
    chat: ChatRef
    # As the person asked: a file name, "all", or None for the recording being read.
    recording: str | None


# Videos still being read, and checks asked for before they were ready. Held in memory like the
# in-process jobs that finish them: after a restart neither exists any more.
class Readings:
    def __init__(self) -> None:
        self._reading: dict[str, Reading] = {}
        self._waiting: dict[str, WaitingCheck] = {}

    def start(self, reading: Reading) -> None:
        self._reading[reading.footage_id] = reading

    # Oldest first.
    def of_line(self, line_id: str) -> tuple[Reading, ...]:
        return tuple(r for r in self._reading.values() if r.line_id == line_id)

    # Asking twice for the same recording still makes one check.
    def wait(self, footage_id: str, check: WaitingCheck) -> None:
        if footage_id in self._reading:
            self._waiting[footage_id] = check

    def finish(self, footage_id: str) -> WaitingCheck | None:
        self._reading.pop(footage_id, None)
        return self._waiting.pop(footage_id, None)
