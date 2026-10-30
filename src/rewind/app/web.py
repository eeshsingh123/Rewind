import asyncio
import hmac
import secrets
import shutil
import tempfile
import time
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import APIRouter, FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from rewind.adapters.storage.local_disk import LocalDiskBlobStore
from rewind.app.upload import MAX_UPLOAD_BYTES
from rewind.contracts import Box
from rewind.core.ask import (
    SPENT,
    Exchange,
    Kind,
    Marked,
    Runs,
    Samples,
    Session,
    Sessions,
    footage_suffix,
)
from rewind.core.ports import BlobStoreError

PAGE = Path(__file__).parent / "static"
# Links are made fresh each time a page loads, so they only need to outlast a long visit.
_MEDIA_TTL_S = 6 * 3600
_MEDIA_TYPES = {".mp4": "video/mp4", ".jpg": "image/jpeg"}
# Smaller than this is a slip of the finger, not a mark.
_MIN_BOX_SIDE = 0.01
_SAMPLES_STILL_WORK = " The samples still work."


class _View(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Created(_View):
    id: str


class SampleView(_View):
    name: str
    title: str
    photo: bool
    picture_url: str


class MomentView(_View):
    start_s: float
    end_s: float
    what: str


class FindingView(_View):
    text: str
    kind: Kind
    proof_url: str | None
    proof_photo: bool
    start_s: float | None
    end_s: float | None


class ExchangeView(_View):
    question: str
    answer: str
    findings: tuple[FindingView, ...]
    steps: tuple[str, ...]
    cost_usd: Decimal
    marked_at_s: float | None


class SessionView(_View):
    id: str
    photo: bool
    duration_s: float
    media_url: str
    summary: str
    moments: tuple[MomentView, ...]
    questions: tuple[str, ...]
    exchanges: tuple[ExchangeView, ...]


class StepView(_View):
    text: str
    done: bool


class LiveView(_View):
    working: bool
    question: str | None
    steps: tuple[StepView, ...]
    problem: str | None


class QuestionBody(_View):
    question: str = Field(min_length=1, max_length=500)
    box: Box | None = None
    at_s: float | None = Field(default=None, ge=0)


def web_router(
    runs: Runs, sessions: Sessions, samples: Samples, blobs: LocalDiskBlobStore
) -> APIRouter:
    router = APIRouter()

    def media_url(key: str) -> str:
        # Relative, so a link works from any address the page was opened at.
        parts = urlsplit(blobs.signed_url(key, _MEDIA_TTL_S))
        return f"{parts.path}?{parts.query}"

    def exchange_view(exchange: Exchange) -> ExchangeView:
        shown: set[str] = set()
        findings: list[FindingView] = []
        for finding in exchange.findings:
            # Several findings can share one proof; it is shown once.
            key = finding.proof_key
            url = media_url(key) if key and key not in shown else None
            if key:
                shown.add(key)
            span = finding.span
            findings.append(
                FindingView(
                    text=finding.text,
                    kind=finding.kind,
                    proof_url=url,
                    proof_photo=bool(key) and PurePosixPath(key or "").suffix == ".jpg",
                    start_s=span.start if span else None,
                    end_s=span.end if span else None,
                )
            )
        return ExchangeView(
            question=exchange.question,
            answer=exchange.answer,
            findings=tuple(findings),
            steps=exchange.steps,
            cost_usd=exchange.cost_usd,
            marked_at_s=exchange.marked.t if exchange.marked else None,
        )

    def session_view(session: Session) -> SessionView:
        look = session.first_look
        return SessionView(
            id=session.id,
            photo=session.photo,
            duration_s=session.footage.duration_s,
            media_url=media_url(session.playback_key),
            summary=look.summary,
            moments=tuple(
                MomentView(start_s=m.span.start, end_s=m.span.end, what=m.what)
                for m in look.moments
            ),
            questions=look.questions,
            exchanges=tuple(exchange_view(e) for e in session.exchanges),
        )

    @router.get("/", include_in_schema=False)
    @router.get("/s/{session_id}", include_in_schema=False)
    async def page() -> FileResponse:
        return FileResponse(PAGE / "index.html", media_type="text/html")

    @router.get("/api/samples")
    async def sample_list() -> list[SampleView]:
        return [
            SampleView(
                name=s.name,
                title=s.title,
                photo=s.photo,
                picture_url=media_url(await samples.picture(s)),
            )
            for s in samples.available()
        ]

    @router.post("/api/samples/{name}", status_code=201)
    async def open_sample(name: str) -> Created:
        sample = samples.named(name)
        if sample is None:
            raise HTTPException(404, "no such sample")
        session_id = _new_id()
        runs.open(session_id, await samples.stored(sample), answer_first=True)
        return Created(id=session_id)

    @router.post("/api/sessions", status_code=201)
    async def upload(file: UploadFile) -> Created:
        if file.size is not None and file.size > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "That file is over 2 GB.")
        session_id = _new_id()
        key = f"uploads/{session_id}/upload{footage_suffix(file.filename, file.content_type)}"
        with tempfile.TemporaryDirectory() as workdir:
            local = Path(workdir) / "upload"
            await asyncio.to_thread(_copy, file, local)
            if local.stat().st_size == 0:
                raise HTTPException(400, "That file is empty.")
            await blobs.write_file(key, local)
        runs.open(session_id, key)
        return Created(id=session_id)

    @router.get("/api/sessions/{session_id}")
    async def session(session_id: str) -> SessionView:
        found = await _get(sessions, session_id)
        if found is None:
            raise HTTPException(404, "no such session")
        return session_view(found)

    @router.get("/api/sessions/{session_id}/live")
    async def live(session_id: str) -> LiveView:
        current = runs.live(session_id)
        if current is None:
            if await _get(sessions, session_id) is None:
                raise HTTPException(404, "no such session")
            return LiveView(working=False, question=None, steps=(), problem=None)
        return LiveView(
            working=current.working,
            question=current.question,
            steps=tuple(StepView(text=s.text, done=s.done) for s in current.steps),
            problem=_shown(current.problem),
        )

    @router.post("/api/sessions/{session_id}/questions", status_code=202)
    async def ask(session_id: str, body: QuestionBody) -> None:
        found = await _get(sessions, session_id)
        if found is None:
            raise HTTPException(404, "no such session")
        if runs.ask(session_id, body.question.strip(), _marked(body, found)) is None:
            raise HTTPException(409, "Still working on the last question.")

    @router.get("/media/{key:path}", include_in_schema=False)
    async def media(
        key: str, expires: Annotated[int, Query()], sig: Annotated[str, Query()]
    ) -> FileResponse:
        if expires < time.time() or not hmac.compare_digest(blobs.signature(key, expires), sig):
            raise HTTPException(403, "this link has expired")
        try:
            path = await blobs.local_path(key)
        except BlobStoreError as error:
            raise HTTPException(404, "no such file") from error
        return FileResponse(path, media_type=_MEDIA_TYPES.get(path.suffix))

    return router


def mount_page(app: FastAPI) -> None:
    app.mount("/static", StaticFiles(directory=PAGE), name="static")


def _new_id() -> str:
    return secrets.token_urlsafe(12)


# An id came from a link: one that was never made simply finds nothing.
async def _get(sessions: Sessions, session_id: str) -> Session | None:
    try:
        return await sessions.get(session_id)
    except BlobStoreError:
        return None


def _marked(body: QuestionBody, session: Session) -> Marked | None:
    box = body.box
    if box is None:
        return None
    if box.x1 - box.x0 < _MIN_BOX_SIDE or box.y1 - box.y0 < _MIN_BOX_SIDE:
        raise HTTPException(422, "That box is too small; drag a larger one.")
    if session.photo:
        return Marked(box=box, t=0.0)
    return Marked(box=box, t=min(body.at_s or 0.0, session.footage.duration_s))


def _shown(problem: str | None) -> str | None:
    return f"{SPENT}{_SAMPLES_STILL_WORK}" if problem == SPENT else problem


def _copy(file: UploadFile, local: Path) -> None:
    with local.open("wb") as destination:
        shutil.copyfileobj(file.file, destination)
