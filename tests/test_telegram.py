import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from rewind.adapters.channels.telegram import TelegramChannel, parse_update
from rewind.adapters.storage.local_disk import LocalDiskBlobStore
from rewind.contracts import Attachment, Button, Reply
from rewind.core.conversation import AttachmentTooLargeError

pytestmark = pytest.mark.anyio

RECORDINGS = Path(__file__).parent / "recordings" / "telegram"


def _recorded(name: str) -> dict[str, Any]:
    return json.loads((RECORDINGS / name).read_text("utf-8"))


def _channel(sent: list[tuple[str, dict[str, Any]]]) -> TelegramChannel:
    def handle(request: httpx.Request) -> httpx.Response:
        sent.append((request.url.path.rsplit("/", 1)[-1], json.loads(request.content)))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    return TelegramChannel(client, "123:token", LocalDiskBlobStore(Path("blobs"), "", b"key"))


def test_a_recorded_video_update_becomes_a_canonical_message() -> None:
    message = parse_update(_recorded("video_message.json"))

    assert message is not None
    assert (message.channel, message.message_id) == ("telegram", "179950836")
    assert (message.chat_id, message.sender_id, message.text) == (
        "100200300",
        "100200300",
        "line video",
    )
    assert message.attachments == (
        Attachment(
            kind="video",
            ref="FILE_FILE_ID",
            mime_type="video/mp4",
            size_bytes=7_737_444,
            filename="pexels_5903898_packages_conveyor_720p.mp4",
        ),
    )


def test_a_recorded_text_update_keeps_its_text() -> None:
    message = parse_update(_recorded("text_message.json"))

    assert message is not None
    assert message.text == "New line: Packing line 2"
    assert message.attachments == ()


async def test_a_reply_renders_bold_safely_with_buttons() -> None:
    sent: list[tuple[str, dict[str, Any]]] = []
    reply = Reply(
        channel="telegram",
        chat_id="42",
        text="Created *Packing line 2* for <file_a_b>.",
        buttons=(Button(label="Check it ($0.66)", value="/confirm abc"),),
    )

    await _channel(sent).send(reply)

    method, body = sent[0]
    assert method == "sendMessage"
    assert body["text"] == "Created <b>Packing line 2</b> for &lt;file_a_b&gt;."
    assert body["reply_markup"] == {
        "inline_keyboard": [[{"text": "Check it ($0.66)", "callback_data": "/confirm abc"}]]
    }


async def test_progress_edits_one_message_in_place() -> None:
    sent: list[tuple[str, dict[str, Any]]] = []
    channel = _channel(sent)

    await channel.show_progress("42", "job-1", "Checking 1 of 2")
    await channel.show_progress("42", "job-1", "Checking 2 of 2")

    assert [method for method, _ in sent] == ["sendMessage", "editMessageText"]
    assert sent[1][1]["message_id"] == 1


async def test_a_video_over_20_mb_is_refused_before_any_call(tmp_path: Path) -> None:
    sent: list[tuple[str, dict[str, Any]]] = []
    big = Attachment(kind="video", ref="file", size_bytes=21 * 1024 * 1024)

    with pytest.raises(AttachmentTooLargeError):
        await _channel(sent).fetch_attachment(big, tmp_path / "clip.mp4")

    assert sent == []
