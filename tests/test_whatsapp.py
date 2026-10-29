import hashlib
import hmac
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from rewind.adapters.channels.whatsapp import WEBHOOK_PATH, webhook_router
from rewind.contracts import CanonicalMessage

# Shaped like Meta's documented webhook; replace with a live recording once WhatsApp is set up.
PAYLOAD = (Path(__file__).parent / "recordings" / "whatsapp" / "video_message.json").read_bytes()
SECRET = "app-secret"


def _client(received: list[CanonicalMessage]) -> TestClient:
    seen: set[str] = set()

    # Stands in for Inbox.receive: claim the id, queue only the first delivery.
    async def receive(message: CanonicalMessage) -> bool:
        if message.message_id in seen:
            return False
        seen.add(message.message_id)
        received.append(message)
        return True

    app = FastAPI()
    app.include_router(webhook_router("verify-me", SECRET, receive))
    return TestClient(app)


def _signed(body: bytes) -> dict[str, str]:
    digest = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    return {"x-hub-signature-256": f"sha256={digest}", "content-type": "application/json"}


def test_the_verify_challenge_needs_the_token() -> None:
    client = _client([])
    query = {"hub.mode": "subscribe", "hub.challenge": "42"}

    good = client.get(WEBHOOK_PATH, params={**query, "hub.verify_token": "verify-me"})
    bad = client.get(WEBHOOK_PATH, params={**query, "hub.verify_token": "wrong"})

    assert (good.status_code, good.text) == (200, "42")
    assert bad.status_code == 403


def test_a_delivery_with_a_bad_signature_is_refused() -> None:
    received: list[CanonicalMessage] = []

    response = _client(received).post(WEBHOOK_PATH, content=PAYLOAD, headers=_signed(b"other"))

    assert response.status_code == 401
    assert received == []


def test_a_repeated_delivery_is_processed_once() -> None:
    received: list[CanonicalMessage] = []
    client = _client(received)

    first = client.post(WEBHOOK_PATH, content=PAYLOAD, headers=_signed(PAYLOAD))
    second = client.post(WEBHOOK_PATH, content=PAYLOAD, headers=_signed(PAYLOAD))

    assert (first.status_code, second.status_code) == (200, 200)
    assert len(received) == 1
    message = received[0]
    assert (message.chat_id, message.attachments[0].ref) == ("15551234567", "MEDIA_ID_1")
    assert message.text == "Every carton needs a label before it leaves the view."
