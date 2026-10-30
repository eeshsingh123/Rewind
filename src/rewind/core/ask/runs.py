import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field

from rewind.core.ask.agent import AskAgent
from rewind.core.ask.session import Exchange, Marked, OnStep, Session, Sessions, Step
from rewind.core.ask.tools import Segmenters
from rewind.core.budget import BudgetExceededError
from rewind.core.ports import LanguageModel, VideoToolsError

logger = logging.getLogger(__name__)

# All web use shares one budget: the page is open to anyone with the link.
WEB_WORKSPACE = "web"
SPENT = "Today's budget is used up, so this can't run right now."
UNREADABLE = "That file couldn't be read as a photo or a video."
FAILED = "Something went wrong. Please try again."

# The paid models for a workspace's session, already behind the budget guard.
type Hands = Callable[[str, str], tuple[LanguageModel, Segmenters]]


# What one session is doing now, for whoever is watching. It lives in memory: finished work is on
# the session itself. A page polls this; a chat is pushed each step and waits for finished.
@dataclass(slots=True)
class Live:
    question: str | None = None
    steps: list[Step] = field(default_factory=list[Step])
    working: bool = True
    problem: str | None = None
    session: Session | None = None
    exchange: Exchange | None = None
    on_step: OnStep | None = None
    finished: asyncio.Event = field(default_factory=asyncio.Event)

    async def add(self, step: Step) -> None:
        self.steps.append(step)
        if self.on_step is not None:
            await self.on_step(step)


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

    def open(
        self,
        session_id: str,
        key: str,
        answer_first: bool = False,
        workspace_id: str = WEB_WORKSPACE,
        on_step: OnStep | None = None,
    ) -> Live:
        live = Live(on_step=on_step)
        work = self._open(session_id, key, live, answer_first, workspace_id)
        self._start(session_id, live, work)
        return live

    # None while the session is still busy.
    def ask(
        self,
        session_id: str,
        question: str,
        marked: Marked | None,
        workspace_id: str = WEB_WORKSPACE,
        on_step: OnStep | None = None,
    ) -> Live | None:
        current = self._live.get(session_id)
        if current is not None and current.working:
            return None
        live = Live(question=question, on_step=on_step)
        self._start(session_id, live, self._ask(session_id, question, marked, live, workspace_id))
        return live

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
        # Someone is waiting on this run, so whatever broke it must end the wait.
        except Exception:
            logger.exception("a run failed")
            live.problem = FAILED
        finally:
            live.working = False
            live.finished.set()

    async def _open(
        self, session_id: str, key: str, live: Live, answer_first: bool, workspace_id: str
    ) -> None:
        model, segmenters = self._hands(workspace_id, session_id)
        async with self._heavy:
            session, _ = await self._sessions.open(session_id, key, model, live.add)
        live.session = session
        questions = session.first_look.questions
        # Only from memory: opening something never spends on a question nobody asked.
        if answer_first and questions and await self._agent.knows(session, questions[0], model):
            live.question = questions[0]
            live.steps.clear()
            await self._agent.ask(session, questions[0], model, segmenters, live.add)

    async def _ask(
        self, session_id: str, question: str, marked: Marked | None, live: Live, workspace_id: str
    ) -> None:
        session = await self._sessions.get(session_id)
        if session is None:
            live.problem = FAILED
            return
        model, segmenters = self._hands(workspace_id, session_id)
        live.exchange = await self._agent.ask(
            session, question, model, segmenters, live.add, marked
        )
