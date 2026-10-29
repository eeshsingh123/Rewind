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


def test_a_delivery_with_a_bad_signature_is_refused() -> None:
    received: list[CanonicalMessage] = []

    response = _client(received).post(WEBHOOK_PATH, content=PAYLOAD, headers=_signed(b"other"))

    assert response.status_code == 401
    assert received == []
