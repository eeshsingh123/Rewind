import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field

from rewind.core.ask.agent import AskAgent
from rewind.core.ask.session import Marked, Sessions, Step
from rewind.core.ask.tools import Segmenters
from rewind.core.budget import BudgetExceededError
from rewind.core.ports import LanguageModel, VideoToolsError

logger = logging.getLogger(__name__)

SPENT = "Today's budget is used up, so this can't run right now. The samples still work."
UNREADABLE = "That file couldn't be read as a photo or a video."
FAILED = "Something went wrong. Please try again."

# The paid models for one session, already behind the budget guard.
type Hands = Callable[[str], tuple[LanguageModel, Segmenters]]


# What one session is doing now, for whoever is watching. It lives in memory: finished work is on
# the session itself.
@dataclass(slots=True)
class Live:
    question: str | None = None
    steps: list[Step] = field(default_factory=list[Step])
    working: bool = True
    problem: str | None = None

    async def add(self, step: Step) -> None:
        self.steps.append(step)


# Opens sessions and asks questions in the background, so closing a page never loses paid work
# and a page opened later can pick up the same steps.
class Runs:
    def __init__(self, sessions: Sessions, agent: AskAgent, hands: Hands) -> None:
        self._sessions = sessions
        self._agent = agent
        self._hands = hands
        self._live: dict[str, Live] = {}
        self._tasks: set[asyncio.Task[None]] = set()
        # 8 GB of RAM, so videos are readied one at a time.
        self._heavy = asyncio.Semaphore(1)

    def live(self, session_id: str) -> Live | None:
        return self._live.get(session_id)

    def open(self, session_id: str, key: str, answer_first: bool = False) -> None:
        live = Live()
        self._start(session_id, live, self._open(session_id, key, live, answer_first))

    # False while the session is still busy.
    def ask(self, session_id: str, question: str, marked: Marked | None) -> bool:
        current = self._live.get(session_id)
        if current is not None and current.working:
            return False
        live = Live(question=question)
        self._start(session_id, live, self._ask(session_id, question, marked, live))
        return True

    async def close(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _start(self, session_id: str, live: Live, work: Coroutine[None, None, None]) -> None:
        self._live[session_id] = live
        task = asyncio.create_task(self._guarded(live, work))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _guarded(self, live: Live, work: Coroutine[None, None, None]) -> None:
        try:
            await work
        except BudgetExceededError:
            live.problem = SPENT
        except VideoToolsError:
            logger.exception("could not read the footage")
            live.problem = UNREADABLE if live.question is None else FAILED
        # A page is waiting on this run, so whatever broke it must end the wait.
        except Exception:
            logger.exception("a run failed")
            live.problem = FAILED
        finally:
            live.working = False

    async def _open(self, session_id: str, key: str, live: Live, answer_first: bool) -> None:
        model, segmenters = self._hands(session_id)
        async with self._heavy:
            session, _ = await self._sessions.open(session_id, key, model, live.add)
        questions = session.first_look.questions
        # Only from memory: opening something never spends on a question nobody asked.
        if answer_first and questions and await self._agent.knows(session, questions[0], model):
            live.question = questions[0]
            live.steps.clear()
            await self._agent.ask(session, questions[0], model, segmenters, live.add)

    async def _ask(self, session_id: str, question: str, marked: Marked | None, live: Live) -> None:
        session = await self._sessions.get(session_id)
        if session is None:
            live.problem = FAILED
            return
        model, segmenters = self._hands(session_id)
        await self._agent.ask(session, question, model, segmenters, live.add, marked)
