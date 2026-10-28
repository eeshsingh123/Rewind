import json
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from rewind.core.ports import SegmenterError


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _Content(_Lenient):
    type: str
    text: str = ""


class _OutputItem(_Lenient):
    type: str
    content: list[_Content] = []


class Usage(_Lenient):
    video_frames_processed: int | None = None
    images_processed: int | None = None


class Response(_Lenient):
    status: str
    model: str
    output: list[_OutputItem] = []
    usage: Usage | None = None

    @property
    def text(self) -> str:
        return "".join(
            part.text
            for item in self.output
            if item.type == "message"
            for part in item.content
            if part.type == "output_text"
        )


class _Uploaded(_Lenient):
    id: str


class MetaApi:
    def __init__(self, client: httpx.AsyncClient, api_key: str, base_url: str) -> None:
        self._client = client
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._base_url = base_url.rstrip("/")

    async def upload_file(self, path: Path, mime: str, expires_after_s: int) -> str:
        with path.open("rb") as file:
            response = await self._post(
                "files",
                data={
                    "purpose": "user_data",
                    "expires_after[anchor]": "created_at",
                    "expires_after[seconds]": str(expires_after_s),
                },
                files={"file": (path.name, file, mime)},
            )
        return _validate(_Uploaded, response.json()).id

    async def create_response(self, payload: dict[str, Any]) -> Response:
        response = await self._post("responses", json={**payload, "stream": False})
        return _completed(_validate(Response, response.json()))

    # Streams so a long video keeps the connection alive. Not background mode: it reports no usage.
    async def stream_response(self, payload: dict[str, Any]) -> Response:
        url = f"{self._base_url}/responses"
        headers = {**self._headers, "Accept": "text/event-stream"}
        body = {**payload, "stream": True}
        try:
            async with self._client.stream("POST", url, headers=headers, json=body) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise SegmenterError(_error_text(response))
                async for line in response.aiter_lines():
                    completed = _completed_event(line)
                    if completed is not None:
                        return completed
        except httpx.HTTPError as error:
            raise SegmenterError(f"Meta API stream failed: {error!r}") from error
        raise SegmenterError("the Meta API stream ended before response.completed")

    async def _post(self, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._client.post(
                f"{self._base_url}/{path}", headers=self._headers, **kwargs
            )
        except httpx.HTTPError as error:
            raise SegmenterError(f"Meta API request failed: {error!r}") from error
        if response.status_code >= 400:
            raise SegmenterError(_error_text(response))
        return response


def _completed_event(line: str) -> Response | None:
    if not line.startswith("data: ") or line == "data: [DONE]":
        return None
    event = json.loads(line.removeprefix("data: "))
    kind = event.get("type")
    if kind == "response.completed":
        return _completed(_validate(Response, event["response"]))
    if kind in ("response.failed", "response.incomplete", "error"):
        raise SegmenterError(f"Meta API {kind}: {json.dumps(event)[:500]}")
    return None


def _completed(response: Response) -> Response:
    if response.status != "completed":
        raise SegmenterError(f"Meta API response ended as {response.status}")
    return response


def _validate[T: BaseModel](model: type[T], data: object) -> T:
    try:
        return model.model_validate(data)
    except ValidationError as error:
        raise SegmenterError(f"unexpected Meta API response: {error}") from error


def _error_text(response: httpx.Response) -> str:
    return f"Meta API {response.status_code}: {response.text[:500]}"
