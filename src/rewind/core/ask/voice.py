import asyncio
import hashlib
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from rewind.core.ports import (
    SPEECH_BYTES_PER_S,
    BlobStore,
    Transcriber,
    TranscriberError,
)


class AudioTooLongError(TranscriberError):
    def __init__(self, max_seconds: int) -> None:
        super().__init__(f"the transcriber takes up to {max_seconds} seconds")
        self.max_seconds = max_seconds


class SpeechMaker(Protocol):
    async def speech(self, source: Path, destination: Path) -> None:
        """Writes the sound as a transcriber's WAV; any phone's audio format goes in."""
        ...


# Turns a voice note into words, once: the same audio is never paid for twice.
class Voice:
    def __init__(
        self,
        blobs: BlobStore,
        speech: SpeechMaker,
        transcriber_for: Callable[[str], Transcriber],
    ) -> None:
        self._blobs = blobs
        self._speech = speech
        self._transcriber_for = transcriber_for

    # The words said, in the language they were said in; empty when nothing could be made out.
    async def hear(self, source: Path, workspace_id: str) -> str:
        transcriber = self._transcriber_for(workspace_id)
        audio = await asyncio.to_thread(source.read_bytes)
        inputs = f"{hashlib.sha256(audio).hexdigest()}|{transcriber.model_version}"
        key = f"ask/heard/{hashlib.sha256(inputs.encode()).hexdigest()[:24]}.txt"
        if await self._blobs.exists(key):
            return (await self._blobs.read(key)).decode()
        with tempfile.TemporaryDirectory() as workdir:
            wav = Path(workdir) / "speech.wav"
            await self._speech.speech(source, wav)
            if wav.stat().st_size / SPEECH_BYTES_PER_S > transcriber.max_seconds:
                raise AudioTooLongError(transcriber.max_seconds)
            transcript = await transcriber.transcribe(wav)
        text = " ".join(transcript.text.split())
        await self._blobs.write(key, text.encode())
        return text
