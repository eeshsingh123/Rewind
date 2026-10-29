import asyncio
import logging
import uuid
from collections.abc import Mapping

from rewind.core.ports import JobHandler, JobPayload, JobRunnerError

logger = logging.getLogger(__name__)


class InProcessJobRunner:
    # A limit per kind keeps heavy jobs one at a time on an 8 GB machine (D-12).
    def __init__(self, limits: Mapping[str, int] | None = None) -> None:
        self._handlers: dict[str, JobHandler] = {}
        self._slots = {kind: asyncio.Semaphore(n) for kind, n in (limits or {}).items()}
        self._tasks: set[asyncio.Task[None]] = set()

    def register(self, kind: str, handler: JobHandler) -> None:
        self._handlers[kind] = handler

    async def submit(self, kind: str, payload: JobPayload) -> str:
        handler = self._handlers.get(kind)
        if handler is None:
            raise JobRunnerError(f"no handler for job kind {kind!r}")
        job_id = uuid.uuid4().hex
        task = asyncio.create_task(self._run(kind, job_id, handler, payload))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job_id

    async def drain(self) -> None:
        while self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _run(self, kind: str, job_id: str, handler: JobHandler, payload: JobPayload) -> None:
        slot = self._slots.get(kind)
        try:
            if slot is None:
                await handler({**payload, "job_id": job_id})
            else:
                async with slot:
                    await handler({**payload, "job_id": job_id})
        except Exception:
            # Nothing is waiting on a background job, so its failure is logged here.
            logger.exception("job %s (%s) failed", job_id, kind)
