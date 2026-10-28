from collections.abc import Awaitable, Callable, Mapping
from typing import Protocol

from pydantic import JsonValue

# JSON only, so a job can cross a process or machine boundary unchanged.
JobPayload = Mapping[str, JsonValue]
JobHandler = Callable[[JobPayload], Awaitable[None]]


class JobRunnerError(Exception):
    pass


class JobRunner(Protocol):
    def register(self, kind: str, handler: JobHandler) -> None: ...

    async def submit(self, kind: str, payload: JobPayload) -> str:
        """Queues a job of a registered kind and returns its id without waiting for it."""
        ...
