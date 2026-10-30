import hashlib
import hmac
import itertools
import json
import re
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from rewind.adapters.channels.telegram import TelegramChannel
from rewind.adapters.channels.whatsapp import (
    GRAPH_BASE,
    WEBHOOK_PATH,
    WhatsAppChannel,
    webhook_router,
)
from rewind.adapters.jobs.in_process import InProcessJobRunner
from rewind.adapters.models.muse_voice import MuseVoice
from rewind.adapters.storage.sql import SqlRepository
from rewind.app.web import web_router
from rewind.contracts import TimeSpan
from rewind.core.ask import TURN_JOB, Chats, ChatSettings, Inbox, Mark, Runs, Samples, Voice
from rewind.core.budget import (
    EPOCH,
    BudgetedTranscriber,
    BudgetExceededError,
    BudgetLimits,
    BudgetScope,
    Ledger,
)
from rewind.core.ports import (
    Completion,
    CompletionRequest,
    Transcriber,
    UserMessage,
)
from tests.fakes import RECORDINGS, FakeSam, ScriptedModel, completion
from tests.test_ask import FIRST_LOOK, PHOTO_LOOK, FakeImageSam, FakeMedia, Flow, _call
from tests.test_web import _answering, _settled, _upload

pytestmark = pytest.mark.anyio

SECRET = "app-secret"
PUBLIC = "https://rewind.test"
QUESTION = "Which cartons have no label?"
NO_LABEL = "The carton 1st from the left at 00:00 has no label."
HINDI_QUESTION = "किन कार्टन पर लेबल नहीं है?"
# Devanagari digits for a time the evidence writes as 00:00.
HINDI_ANSWER = "बाएँ से पहले कार्टन पर ००:०० पर लेबल नहीं है।"  # noqa: RUF001
# Delivery ids are never reused, as a repeated one is dropped as a retry.
_IDS = itertools.count(1000)
_CAPTION = re.compile(rb'name="caption"\r\n\r\n(.*?)\r\n--', re.DOTALL)


# What the chat showed: "text", "edit", "video", "photo", "typing" or "read".
@dataclass
class Shown:
    kind: str
    text: str = ""
    # (label, value) of each choice offered.
    choices: tuple[tuple[str, str], ...] = ()


def _recording(path: str) -> dict[str, Any]:
    return json.loads((RECORDINGS / path).read_text("utf-8"))


class TelegramPhone:
    name = "telegram"

    def __init__(self, footage: dict[str, bytes]) -> None:
        self.shown: list[Shown] = []
        self.acknowledged = 0
        self.fetched: list[str] = []
        self._footage = footage
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self._handle))

    def attach(self, channel: TelegramChannel, inbox: Inbox) -> None:
        self.channel, self._inbox = channel, inbox

    async def send(self, recording: str, **changes: Any) -> None:
        update = _recording(f"telegram/{recording}.json")
        update["update_id"] = next(_IDS)
        update["message"].update(changes)
        await self.channel._dispatch(update, self._inbox.receive)

    async def tap(self, value: str) -> None:
        update = _recording("telegram/tap.json")
        update["update_id"] = next(_IDS)
        update["callback_query"]["data"] = value
        await self.channel._dispatch(update, self._inbox.receive)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if "/file/" in request.url.path:
            return httpx.Response(200, content=self._footage["file"])
        method = request.url.path.rpartition("/")[2]
        if method == "getFile":
            ref = json.loads(request.content)["file_id"]
            self.fetched.append(ref)
            return _ok({"file_path": "files/1"})
        if method in ("sendVideo", "sendPhoto"):
            found = _CAPTION.search(request.content)
            caption = found.group(1).decode() if found else ""
            self.shown.append(Shown("video" if method == "sendVideo" else "photo", caption))
        elif method == "answerCallbackQuery":
            self.acknowledged += 1
        elif method == "sendChatAction":
            self.shown.append(Shown("typing"))
        else:
            body = json.loads(request.content)
            rows = body.get("reply_markup", {}).get("inline_keyboard", [])
            choices = tuple((b["text"], b["callback_data"]) for row in rows for b in row)
            kind = "edit" if method == "editMessageText" else "text"
            self.shown.append(Shown(kind, body["text"], choices))
        return _ok({"message_id": next(_IDS)})


class WhatsAppPhone:
    name = "whatsapp"

    def __init__(self, footage: dict[str, bytes], media_bytes: int = 1000) -> None:
        self.shown: list[Shown] = []
        self.fetched: list[str] = []
        self.media_bytes = media_bytes
        self._footage = footage
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(self._handle))

    def attach(self, inbox: Inbox) -> None:
        app = FastAPI()
        app.include_router(webhook_router("verify", SECRET, inbox.receive))
        self._webhook = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )

    async def send(self, recording: str, **changes: Any) -> None:
        webhook = _recording(f"whatsapp/{recording}.json")
        message = webhook["entry"][0]["changes"][0]["value"]["messages"][0]
        message["id"] = f"wamid.{next(_IDS)}"
        message.update(changes)
        await self._deliver(webhook)

    async def tap(self, value: str) -> None:
        webhook = _recording("whatsapp/list_reply.json")
        message = webhook["entry"][0]["changes"][0]["value"]["messages"][0]
        message["id"] = f"wamid.{next(_IDS)}"
        message["interactive"]["list_reply"]["id"] = value
        await self._deliver(webhook)

    async def _deliver(self, webhook: dict[str, Any]) -> None:
        body = json.dumps(webhook).encode()
        digest = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
        headers = {"x-hub-signature-256": f"sha256={digest}"}
        response = await self._webhook.post(WEBHOOK_PATH, content=body, headers=headers)
        assert response.status_code == 200

    def _handle(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request.method == "GET":
            if url.startswith(GRAPH_BASE):
                self.fetched.append(url.rpartition("/")[2])
                info = {"url": "https://lookaside.test/file", "file_size": self.media_bytes}
                return httpx.Response(200, json=info)
            return httpx.Response(200, content=self._footage["file"])
        if url.endswith("/media"):
            return httpx.Response(200, json={"id": "media-1"})
        body = json.loads(request.content)
        if body.get("status") == "read":
            assert body["typing_indicator"] == {"type": "text"}
            self.shown.append(Shown("typing"))
        elif body["type"] == "text":
            self.shown.append(Shown("text", body["text"]["body"]))
        elif body["type"] == "interactive":
            interactive = body["interactive"]
            assert interactive["type"] == "list"
            rows = interactive["action"]["sections"][0]["rows"]
            assert all(len(row["title"]) <= 24 for row in rows)
            assert all(len(row.get("description", "")) <= 72 for row in rows)
            label = [f"{r['title']} {r.get('description', '')}".strip() for r in rows]
            choices = tuple(zip(label, [r["id"] for r in rows], strict=True))
            self.shown.append(Shown("text", interactive["body"]["text"], choices))
        else:
            kind = body["type"]
            self.shown.append(Shown(kind, body[kind].get("caption", "")))
        return httpx.Response(200, json={"messages": [{"id": "sent"}]})


def _ok(result: object) -> httpx.Response:
    return httpx.Response(200, json={"ok": True, "result": result})


class HugeProofs(FakeMedia):
    async def make_clip(
        self, source: Path, destination: Path, span: TimeSpan, marks: Sequence[Mark]
    ) -> None:
        with destination.open("wb") as file:
            file.seek(60_000_000)
            file.write(b"\0")


@dataclass
class Speech:
    heard: str = HINDI_QUESTION
    requests: list[httpx.Request] = field(default_factory=list[httpx.Request])

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={
            "sessionId": "s", "transcript": self.heard, "audioDurationMs": 2600, "turns": [],
        })  # fmt: skip


@dataclass
class Desk:
    phone: Any
    flow: Flow
    model: ScriptedModel
    repository: SqlRepository
    speech: Speech
    jobs: InProcessJobRunner
    runs: Runs

    async def send(self, recording: str, **changes: Any) -> None:
        await self.phone.send(recording, **changes)
        await self.jobs.drain()

    async def tap(self, value: str) -> None:
        await self.phone.tap(value)
        await self.jobs.drain()

    def said(self) -> list[Shown]:
        return [s for s in self.phone.shown if s.kind != "typing"]

    def last(self) -> Shown:
        return self.said()[-1]


@asynccontextmanager
async def desk(
    tmp_path: Path,
    channel: str,
    model: ScriptedModel,
    sam: FakeSam | None = None,
    media: FakeMedia | None = None,
    public_url: str = PUBLIC,
    daily_cap: Decimal = Decimal(5),
    media_bytes: int = 1000,
    flow: Flow | None = None,
    repository: SqlRepository | None = None,
) -> AsyncIterator[Desk]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    flow = flow or Flow(tmp_path / "blobs", sam or FakeSam(), media or FakeMedia())
    fresh = repository is None
    if repository is None:
        repository = SqlRepository.from_url(f"sqlite+aiosqlite:///{tmp_path / 'rewind.db'}")
        await repository.create_tables()
    footage = {"file": b"sent from a phone"}
    jobs = InProcessJobRunner()
    inbox = Inbox(repository, jobs)
    phone: Any
    adapter: Any
    if channel == "telegram":
        phone = TelegramPhone(footage)
        adapter = TelegramChannel(phone.client, "token", flow.blobs)
        phone.attach(adapter, inbox)
    else:
        phone = WhatsAppPhone(footage, media_bytes)
        adapter = WhatsAppChannel(phone.client, "token", "1234567890", flow.blobs)
        phone.attach(inbox)
    speech = Speech()
    speech_client = httpx.AsyncClient(transport=httpx.MockTransport(speech.handle))
    ledger = Ledger(repository, BudgetLimits(daily_cap, Decimal(100)))

    def transcriber(workspace_id: str) -> Transcriber:
        voice = MuseVoice(speech_client, "key", "https://api.test/v1")
        return BudgetedTranscriber(voice, ledger, BudgetScope(workspace_id))

    runs = Runs(flow.sessions, flow.agent, lambda *_: (model, flow.segmenters))
    chats = Chats(
        runs, flow.sessions, repository, flow.blobs,
        Voice(flow.blobs, flow.media, transcriber),
        {channel: adapter}, ChatSettings(frozenset(), public_url, 60.0, 0.0),
    )  # fmt: skip
    jobs.register(TURN_JOB, chats.handle_job)
    yield Desk(phone, flow, model, repository, speech, jobs, runs)
    await runs.close()
    if fresh:
        await repository.close()


both = pytest.mark.parametrize("channel", ["telegram", "whatsapp"])


# A phone's whole flow, through each channel's real parsing and sending: footage, first look, a
# tap on a suggested question, live steps, and one answer with its proof and timeline link. The
# chat remembers its footage over a restart.
@both
async def test_footage_first_look_tap_and_answer_with_proof(tmp_path: Path, channel: str) -> None:
    model = ScriptedModel([
        completion(json.dumps(FIRST_LOOK)),
        *_answering({"text": NO_LABEL, "evidence": ["e4"]}),
        completion("", _call("answer", {"claims": [{"text": "Nothing else stands out here.",
                                                    "evidence": ["e1"]}]})),
    ])  # fmt: skip
    async with desk(tmp_path, channel, model) as run:
        await run.send("video_message" if channel == "whatsapp" else "video")
        look = run.last()
        assert FIRST_LOOK["summary"] in look.text
        assert "A carton without a label passes." in look.text
        assert "00:01 to 00:02" in look.text
        [(label, value)] = look.choices
        assert label == QUESTION
        assert len(value.encode()) <= 64
        session_id = value.split(":")[1]

        await run.tap(value)

        answer = run.last()
        assert answer.kind == "video"
        assert NO_LABEL in answer.text
        assert "cost " in answer.text.lower()
        assert f"Open the full timeline: {PUBLIC}/s/{session_id}" in answer.text
        saved = await run.flow.sessions.get(session_id)
        assert saved is not None
        assert [e.question for e in saved.exchanges] == [QUESTION]
        # Steps show live where a message can be edited; elsewhere one line, then the answer.
        steps = [s for s in run.said() if QUESTION in s.text.split("\n")[0]]
        if channel == "telegram":
            assert run.phone.acknowledged == 1
            assert any("✓ Found 2 cartons" in s.text for s in steps)
            assert "⏳" not in steps[-1].text
        else:
            assert len(steps) == 1
        assert any(s.kind == "typing" for s in run.phone.shown)

        flow, repository = run.flow, run.repository
        async with desk(tmp_path, channel, model, flow=flow, repository=repository) as restarted:
            await restarted.send("text_message" if channel == "whatsapp" else "text",
                                 **_text(channel, "Anything else?"))  # fmt: skip
            assert "Nothing else stands out here." in restarted.last().text
            assert "Anything else?" in restarted.said()[0].text
            assert restarted.said()[0].kind == "text"


def _text(channel: str, text: str) -> dict[str, Any]:
    return {"text": {"body": text}} if channel == "whatsapp" else {"text": text}


# A photo's caption is the first question, and its proof is a picture. A HEIC sent as a plain file
# is read as a photo too.
@both
async def test_a_photo_with_a_caption_is_answered_with_a_picture(
    tmp_path: Path, channel: str
) -> None:
    no_label = "The 2nd box from the left has no label on top."
    model = ScriptedModel([
        completion(json.dumps(PHOTO_LOOK)),
        completion("", _call("find", {"thing": "box"})),
        completion("", _call("look", {"question": "Is there a label on top?", "thing": "box"})),
        completion(json.dumps({"tiles": [{"tile": "1", "answer": "has a label"},
                                         {"tile": "2", "answer": "no label"},
                                         {"tile": "3", "answer": "has a label"}]})),
        completion("", _call("answer", {"claims": [{"text": no_label, "evidence": ["e5"]}]})),
    ])  # fmt: skip
    async with desk(tmp_path, channel, model, sam=FakeImageSam()) as run:
        await run.send("image_message" if channel == "whatsapp" else "photo")

        assert no_label in run.last().text
        assert run.last().kind == ("photo" if channel == "telegram" else "image")
        if channel == "telegram":
            assert run.phone.fetched == ["tg-large"]
        assert any("The box on the right looks open." in s.text for s in run.said())

        await run.send("heic_file_message" if channel == "whatsapp" else "heic_file")
        assert PHOTO_LOOK["summary"] in run.last().text
        assert list((run.flow.tmp_path / "uploads").rglob("*.heic"))


# The same question on any channel gives the same answer, and only the first one pays.
async def test_the_same_question_gives_the_same_answer_on_every_channel(tmp_path: Path) -> None:
    model = ScriptedModel([
        completion(json.dumps(FIRST_LOOK)),
        *_answering({"text": NO_LABEL, "evidence": ["e4"]}),
    ])  # fmt: skip
    async with desk(tmp_path, "telegram", model) as telegram:
        await telegram.send("video")
        await telegram.tap(telegram.last().choices[0][1])
        paid = len(model.requests)
        first = telegram.last().text.split("\n")[0]

        flow, repository = telegram.flow, telegram.repository
        async with desk(tmp_path, "whatsapp", model, flow=flow, repository=repository) as whatsapp:
            await whatsapp.send("video_message")
            await whatsapp.tap(whatsapp.last().choices[0][1])
            answer = whatsapp.last()
            assert answer.text.split("\n")[0] == first
            assert "No cost: already answered" in answer.text

        runs = Runs(flow.sessions, flow.agent, lambda *_: (model, flow.segmenters))
        app = FastAPI()
        shelf = Samples((), flow.blobs, flow.media, flow.media)
        app.include_router(web_router(runs, flow.sessions, shelf, flow.blobs))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as web:
            session_id = await _upload(web, "phone.mp4", b"sent from a phone")
            await _settled(web, session_id)
            await web.post(f"/api/sessions/{session_id}/questions", json={"question": QUESTION})
            await _settled(web, session_id)
            [exchange] = (await web.get(f"/api/sessions/{session_id}")).json()["exchanges"]
        assert exchange["answer"] == first
        assert len(model.requests) == paid
        await runs.close()


# A voice note is turned into the question by Meta's speech-to-text, never sent to Muse, and the
# answer comes back in its language. The same note is not paid for twice, and with no footage
# yet, nothing is paid at all.
@both
async def test_a_voice_note_is_answered_in_its_language(tmp_path: Path, channel: str) -> None:
    media = FakeMedia()
    model = ScriptedModel([
        completion(json.dumps(FIRST_LOOK)),
        *_answering({"text": HINDI_ANSWER, "evidence": ["e4"]}),
    ])  # fmt: skip
    voice = "voice_message" if channel == "whatsapp" else "voice"
    async with desk(tmp_path, channel, model, media=media) as run:
        await run.send(voice)
        assert "photo or a video first" in run.last().text
        assert run.speech.requests == []

        await run.send("video_message" if channel == "whatsapp" else "video")
        await run.send(voice)

        [sent] = run.speech.requests
        assert str(sent.url) == "https://api.test/v1/asr/transcribe"
        assert b'"model": "muse-voice-transcribe-1.0"' in sent.content
        assert media.converted == 1
        asked = model.requests[-1].messages
        assert HINDI_QUESTION in [m.text for m in asked if isinstance(m, UserMessage)]
        assert not any(isinstance(m, UserMessage) and m.video_key for m in asked)
        answer = run.last()
        assert HINDI_ANSWER in answer.text
        assert answer.kind == "video"
        assert any(HINDI_QUESTION in s.text for s in run.said())
        assert await run.repository.spent_usd(EPOCH) == Decimal("0.00010")

        await run.send(voice)
        assert len(run.speech.requests) == 1


# A file too big for the channel gets the web page's address, or plain advice when the address
# would not open on a phone; a proof too big to send still leaves the answer and the link.
@both
async def test_what_is_too_big_never_fails_silently(tmp_path: Path, channel: str) -> None:
    big = "video_message" if channel == "whatsapp" else "big_video"
    model = ScriptedModel([])
    async with desk(tmp_path / "public", channel, model, media_bytes=20_000_000) as run:
        await run.send(big)
        assert "too big for this chat" in run.last().text
        assert f"web page instead: {PUBLIC}" in run.last().text
        assert model.requests == []

    async with desk(
        tmp_path / "local", channel, model, media_bytes=20_000_000,
        public_url="http://127.0.0.1:8000",
    ) as run:  # fmt: skip
        await run.send(big)
        assert "shorter clip" in run.last().text
        assert "127.0.0.1" not in run.last().text

    scripted = ScriptedModel([
        completion(json.dumps(FIRST_LOOK)),
        *_answering({"text": NO_LABEL, "evidence": ["e4"]}),
    ])  # fmt: skip
    async with desk(tmp_path / "proof", channel, scripted, media=HugeProofs()) as run:
        await run.send("video_message" if channel == "whatsapp" else "video")
        await run.tap(run.last().choices[0][1])
        answer = run.last()
        assert answer.kind in ("text", "edit")
        assert NO_LABEL in answer.text
        assert "too big to send here" in answer.text
        assert "Open the full timeline" in answer.text


class SpentModel(ScriptedModel):
    async def complete(self, request: CompletionRequest) -> Completion:
        if self.replies:
            return await super().complete(request)
        raise BudgetExceededError("daily cap")


# Spent is said in one plain line, with no paid call: for a question and for a voice note. Fixed
# commands cost nothing, and a command left in an old menu never reaches the agent.
@both
async def test_a_spent_budget_and_commands(tmp_path: Path, channel: str) -> None:
    text = "text_message" if channel == "whatsapp" else "text"
    voice = "voice_message" if channel == "whatsapp" else "voice"
    model = SpentModel([completion(json.dumps(FIRST_LOOK))])
    async with desk(tmp_path, channel, model, daily_cap=Decimal(0)) as run:
        await run.send(text, **_text(channel, "/start"))
        assert "Hi, I'm *Rewind*" in run.last().text.replace("<b>", "*").replace("</b>", "*")
        await run.send(text, **_text(channel, "/rules"))
        assert "Send me a photo or a video" in run.last().text
        assert model.requests == []

        await run.send("video_message" if channel == "whatsapp" else "video")
        await run.send(text)
        assert run.last().text == "Today's budget is used up, so this can't run right now."

        await run.send(voice)
        assert run.last().text == "Today's budget is used up, so this can't run right now."
        assert run.speech.requests == []
