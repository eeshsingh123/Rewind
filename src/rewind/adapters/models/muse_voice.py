import asyncio
import json
from decimal import Decimal
from pathlib import Path

import httpx
from pydantic import BaseModel, Field, ValidationError

from rewind.adapters.models.muse import MAX_ATTEMPTS, PROVIDER, RETRYABLE
from rewind.contracts import Charge
from rewind.core.ports import TranscriberError, Transcript

MODEL = "muse-voice-transcribe-1.0"
# $0.18 per hour; billing rounds down to whole seconds, and failed requests are not billed.
USD_PER_SECOND = Decimal("0.00005")
# The batch endpoint's limits: 10 minutes of audio, 32 MB of request.
MAX_SECONDS = 600
REQUEST_TIMEOUT_S = 120.0


class _Transcribed(BaseModel):
    transcript: str
    duration_ms: int = Field(alias="audioDurationMs")


# Meta's batch speech-to-text. The language is left out, so the model detects it.
class MuseVoice:
    def __init__(self, client: httpx.AsyncClient, api_key: str, base_url: str) -> None:
        self._client = client
        self._api_key = api_key
        self._url = f"{base_url.rstrip('/')}/asr/transcribe"

    @property
    def model_version(self) -> str:
        return MODEL

    @property
    def max_seconds(self) -> int:
        return MAX_SECONDS

    def estimate(self, seconds: int) -> tuple[Charge, ...]:
        return (_charge(seconds),)

    async def transcribe(self, source: Path) -> Transcript:
        audio = await asyncio.to_thread(source.read_bytes)
        response = await self._post(audio)
        try:
            found = _Transcribed.model_validate_json(response.content)
        except ValidationError as error:
            raise TranscriberError(f"unexpected transcription response: {error}") from error
        return Transcript(found.transcript, (_charge(found.duration_ms // 1000),))

    async def _post(self, audio: bytes) -> httpx.Response:
        settings = json.dumps({"model": MODEL, "audioEncoding": "WAV"})
        attempt = 0
        while True:
            try:
                response = await self._client.post(
                    self._url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    files={
                        "request": (None, settings, "application/json"),
                        "audio": ("speech.wav", audio, "audio/wav"),
                    },
                    timeout=REQUEST_TIMEOUT_S,
                )
            except httpx.HTTPError as error:
                message = f"transcription request failed: {type(error).__name__}"
                raise TranscriberError(message) from error
            if response.is_success:
                return response
            attempt += 1
            if response.status_code not in RETRYABLE or attempt == MAX_ATTEMPTS:
                raise TranscriberError(f"transcription returned {response.status_code}")
            await asyncio.sleep(2 ** (attempt - 1))


def _charge(seconds: int) -> Charge:
    return Charge(
        provider=PROVIDER,
        model=MODEL,
        unit="audio_second",
        quantity=seconds,
        usd=seconds * USD_PER_SECOND,
    )
