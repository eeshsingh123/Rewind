from pathlib import Path
from typing import Protocol


class BlobStoreError(Exception):
    pass


class BlobStore(Protocol):
    async def write(self, key: str, data: bytes) -> None: ...

    async def write_file(self, key: str, source: Path) -> None: ...

    async def read(self, key: str) -> bytes: ...

    async def local_path(self, key: str) -> Path:
        """A readable local file with the blob's content, fetched first if the store is remote."""
        ...

    async def exists(self, key: str) -> bool: ...

    def signed_url(self, key: str, expires_in_s: int) -> str: ...
