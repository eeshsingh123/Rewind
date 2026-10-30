import time

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from rewind.core.ports import BlobStore

# An uploaded file is reused for every call only while it has at least this long to live.
_REUSE_MARGIN_S = 600


class MetaFilesError(Exception):
    pass


class _Uploaded(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str


# One upload per video, shared by SAM and Muse: both reference it by file id.
class MetaFiles:
    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        base_url: str,
        blobs: BlobStore,
        ttl_s: int = 86_400,
    ) -> None:
        self._client = client
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._base_url = base_url.rstrip("/")
        self._blobs = blobs
        self._ttl_s = ttl_s
        self._uploads: dict[str, tuple[str, float]] = {}

    async def file_id(self, key: str) -> str:
        cached = self._uploads.get(key)
        now = time.monotonic()
        if cached is not None and cached[1] - now > _REUSE_MARGIN_S:
            return cached[0]
        file_id = await self._upload(key)
        self._uploads[key] = (file_id, now + self._ttl_s)
        return file_id

    async def _upload(self, key: str) -> str:
        path = await self._blobs.local_path(key)
        with path.open("rb") as file:
            try:
                response = await self._client.post(
                    f"{self._base_url}/files",
                    headers=self._headers,
                    data={
                        "purpose": "user_data",
                        "expires_after[anchor]": "created_at",
                        "expires_after[seconds]": str(self._ttl_s),
                    },
                    files={"file": (path.name, file, "video/mp4")},
                )
            except httpx.HTTPError as error:
                raise MetaFilesError(f"Meta file upload failed: {error!r}") from error
        if response.status_code >= 400:
            raise MetaFilesError(f"Meta API {response.status_code}: {response.text[:500]}")
        try:
            return _Uploaded.model_validate(response.json()).id
        except ValidationError as error:
            raise MetaFilesError(f"unexpected Meta upload response: {error}") from error
