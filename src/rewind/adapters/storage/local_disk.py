import asyncio
import hashlib
import hmac
import shutil
import time
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from rewind.core.ports import BlobStoreError


class LocalDiskBlobStore:
    def __init__(self, root: Path, public_base_url: str, signing_key: bytes) -> None:
        self._root = root.resolve()
        self._base_url = public_base_url.rstrip("/")
        self._signing_key = signing_key

    async def write(self, key: str, data: bytes) -> None:
        await asyncio.to_thread(self._write, self._path(key), data)

    async def write_file(self, key: str, source: Path) -> None:
        await asyncio.to_thread(self._copy, source, self._path(key))

    async def read(self, key: str) -> bytes:
        path = self._path(key)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError as error:
            raise BlobStoreError(f"no blob {key}") from error

    async def local_path(self, key: str) -> Path:
        path = self._path(key)
        if not path.is_file():
            raise BlobStoreError(f"no blob {key}")
        return path

    async def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def signed_url(self, key: str, expires_in_s: int) -> str:
        self._path(key)
        expires = int(time.time()) + expires_in_s
        signature = self.signature(key, expires)
        return f"{self._base_url}/media/{quote(key)}?expires={expires}&sig={signature}"

    def signature(self, key: str, expires: int) -> str:
        message = f"{key}\n{expires}".encode()
        return hmac.new(self._signing_key, message, hashlib.sha256).hexdigest()

    def _path(self, key: str) -> Path:
        parts = PurePosixPath(key).parts
        unsafe = any(part in ("..", ".") or ":" in part or "\\" in part for part in parts)
        if not parts or key.startswith("/") or unsafe:
            raise BlobStoreError(f"invalid blob key {key!r}")
        return self._root.joinpath(*parts)

    @staticmethod
    def _write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write then rename, so a reader never sees half a blob.
        partial = path.with_name(path.name + ".partial")
        partial.write_bytes(data)
        partial.replace(path)

    @staticmethod
    def _copy(source: Path, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        shutil.copyfile(source, partial)
        partial.replace(path)
