from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rewind.contracts import Charge

# What a transcriber is given: mono 16-bit PCM WAV at 24 kHz.
SPEECH_RATE_HZ = 24_000
SPEECH_BYTES_PER_S = SPEECH_RATE_HZ * 2


class TranscriberError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    charges: tuple[Charge, ...]


class Transcriber(Protocol):
    @property
    def model_version(self) -> str: ...

    @property
    def max_seconds(self) -> int: ...

    def estimate(self, seconds: int) -> tuple[Charge, ...]: ...

    async def transcribe(self, source: Path) -> Transcript: ...
