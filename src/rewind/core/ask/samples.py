import tempfile
from dataclasses import dataclass
from pathlib import Path

from rewind.core.ask.session import PhotoReader, is_photo
from rewind.core.perception import StillGrabber
from rewind.core.ports import BlobStore

# Past any fade-in at the start of a clip.
_PICTURE_AT_S = 1.0


@dataclass(frozen=True, slots=True)
class Sample:
    name: str
    title: str
    source: Path

    @property
    def photo(self) -> bool:
        return is_photo(self.source.name)


# Footage a visitor can tap instead of uploading. A sample whose file is missing is left out.
class Samples:
    def __init__(
        self,
        samples: tuple[Sample, ...],
        blobs: BlobStore,
        stills: StillGrabber,
        photos: PhotoReader,
    ) -> None:
        self._samples = samples
        self._blobs = blobs
        self._stills = stills
        self._photos = photos

    def available(self) -> tuple[Sample, ...]:
        return tuple(s for s in self._samples if s.source.is_file())

    def named(self, name: str) -> Sample | None:
        return next((s for s in self.available() if s.name == name), None)

    # The blob a session opens: the same content every time, so it costs nothing after the first.
    async def stored(self, sample: Sample) -> str:
        key = f"{_stem(sample)}{sample.source.suffix.lower()}"
        if not await self._blobs.exists(key):
            await self._blobs.write_file(key, sample.source)
        return key

    async def picture(self, sample: Sample) -> str:
        key = f"{_stem(sample)}.picture.jpg"
        if await self._blobs.exists(key):
            return key
        with tempfile.TemporaryDirectory() as workdir:
            destination = Path(workdir) / "picture.jpg"
            if sample.photo:
                await self._photos.upright(sample.source, destination)
            else:
                await self._stills.still(sample.source, _PICTURE_AT_S, destination)
            await self._blobs.write_file(key, destination)
        return key


# Changes with the file, so a swapped sample is picked up and one still being read is left alone.
def _stem(sample: Sample) -> str:
    return f"samples/{sample.name}-{sample.source.stat().st_mtime_ns}"
