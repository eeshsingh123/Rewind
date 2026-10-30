import asyncio
import base64
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from email.utils import parsedate_to_datetime

import httpx
from pydantic import BaseModel, Field, JsonValue, ValidationError

from rewind.adapters.meta_files import MetaFiles, MetaFilesError
from rewind.contracts import Charge
from rewind.core.ports import (
    AssistantMessage,
    BlobStore,
    ChatMessage,
    Completion,
    CompletionRequest,
    LanguageModelError,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

BASE_URL = "https://api.meta.ai/v1"
PROVIDER = "meta"
REQUEST_TIMEOUT_S = 120.0
MAX_ATTEMPTS = 4
MAX_WAIT_S = 60.0
# 429 is the 150 RPM limit; 500 and 503 are transient on this API and safe to replay.
RETRYABLE = frozenset({429, 500, 502, 503})
PER_MILLION = Decimal(1_000_000)


@dataclass(frozen=True, slots=True)
class TokenPrices:
    # USD per 1M tokens. Reasoning tokens are billed as output.
    input: Decimal
    cached_input: Decimal
    output: Decimal


PRICES = {
    "muse-spark-1.3-contributor": TokenPrices(Decimal("0.10"), Decimal("0.002"), Decimal("0.20")),
    "muse-spark-1.3": TokenPrices(Decimal("1.25"), Decimal("0.15"), Decimal("4.25")),
}


class MuseLanguageModel:
    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        model: str,
        images: BlobStore,
        files: MetaFiles | None = None,
    ) -> None:
        if model not in PRICES:
            raise ValueError(f"no prices known for Muse model {model}")
        self._client = client
        self._api_key = api_key
        self._model = model
        self._prices = PRICES[model]
        self._images = images
        self._files = files

    @property
    def model_version(self) -> str:
        return self._model

    async def complete(self, request: CompletionRequest) -> Completion:
        if any(isinstance(m, UserMessage) and m.video_key for m in request.messages):
            return await self._watch(request)
        body = await self._body(request)
        response = await self._post("chat/completions", body)
        try:
            reply = _ChatResponse.model_validate_json(response.content)
        except ValidationError as error:
            raise LanguageModelError(f"unexpected Muse response: {error}") from error
        message = reply.choices[0].message
        return Completion(
            text=message.content or "",
            tool_calls=tuple(
                ToolCall(id=call.id, name=call.function.name, arguments=call.function.arguments)
                for call in message.tool_calls or ()
            ),
            charges=self._charges(reply.usage),
        )

    async def _body(self, request: CompletionRequest) -> dict[str, JsonValue]:
        body: dict[str, JsonValue] = {
            "model": self._model,
            "messages": [await self._message(message) for message in request.messages],
            "max_completion_tokens": request.max_output_tokens,
            "reasoning_effort": request.effort,
        }
        cache_key = _prompt_cache_key(request.messages)
        if cache_key is not None:
            body["prompt_cache_key"] = cache_key
        if request.response_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": dict(request.response_schema)},
            }
        if request.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": dict(tool.parameters),
                    },
                }
                for tool in request.tools
            ]
        return body

    async def _message(self, message: ChatMessage) -> dict[str, JsonValue]:
        match message:
            case SystemMessage(text=text):
                return {"role": "system", "content": text}
            case UserMessage(text=text, image_keys=()):
                return {"role": "user", "content": text}
            case UserMessage(text=text, image_keys=keys):
                parts: list[JsonValue] = [{"type": "text", "text": text}]
                for key in keys:
                    parts.append(
                        {"type": "image_url", "image_url": {"url": await self._image(key)}}
                    )
                return {"role": "user", "content": parts}
            case AssistantMessage(text=text, tool_calls=calls):
                turn: dict[str, JsonValue] = {"role": "assistant", "content": text or None}
                if calls:
                    turn["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.arguments},
                        }
                        for call in calls
                    ]
                return turn
            case ToolResultMessage(tool_call_id=call_id, text=text):
                return {"role": "tool", "tool_call_id": call_id, "content": text}

    # Video is read only through the Responses API, by the file id of an upload.
    async def _watch(self, request: CompletionRequest) -> Completion:
        if request.tools:
            raise LanguageModelError("a request with a video cannot take tools")
        body: dict[str, JsonValue] = {
            "model": self._model,
            "input": [await self._input(message) for message in request.messages],
            "max_output_tokens": request.max_output_tokens,
            "reasoning": {"effort": request.effort},
        }
        cache_key = _prompt_cache_key(request.messages)
        if cache_key is not None:
            body["prompt_cache_key"] = cache_key
        if request.response_schema is not None:
            body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "reply",
                    "strict": True,
                    "schema": dict(request.response_schema),
                }
            }
        response = await self._post("responses", body)
        try:
            reply = _Response.model_validate_json(response.content)
        except ValidationError as error:
            raise LanguageModelError(f"unexpected Muse response: {error}") from error
        if reply.status != "completed":
            raise LanguageModelError(f"Muse response ended as {reply.status}")
        text = "".join(
            part.text
            for item in reply.output
            if item.type == "message"
            for part in item.content
            if part.type == "output_text"
        )
        usage = reply.usage
        cached = usage.input_tokens_details.cached_tokens if usage.input_tokens_details else 0
        details = usage.output_tokens_details
        reasoning = details.reasoning_tokens if details else 0
        charges = self._priced(usage.input_tokens, cached, usage.output_tokens, reasoning)
        return Completion(text=text, tool_calls=(), charges=charges)

    async def _input(self, message: ChatMessage) -> dict[str, JsonValue]:
        match message:
            case SystemMessage(text=text):
                content: list[JsonValue] = [{"type": "input_text", "text": text}]
                return {"type": "message", "role": "system", "content": content}
            case UserMessage(text=text, image_keys=keys, video_key=video_key):
                content = [{"type": "input_text", "text": text}]
                for key in keys:
                    content.append({"type": "input_image", "image_url": await self._image(key)})
                if video_key:
                    content.append({"type": "input_video", "file_id": await self._file(video_key)})
                return {"type": "message", "role": "user", "content": content}
            case _:
                raise LanguageModelError("a request with a video takes system and user text only")

    async def _file(self, key: str) -> str:
        if self._files is None:
            raise LanguageModelError("this Muse model was built without file uploads")
        try:
            return await self._files.file_id(key)
        except MetaFilesError as error:
            raise LanguageModelError(str(error)) from error

    async def _image(self, key: str) -> str:
        data = await self._images.read(key)
        return f"data:{_image_type(data)};base64,{base64.b64encode(data).decode()}"

    async def _post(self, path: str, body: dict[str, JsonValue]) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        attempt = 0
        while True:
            try:
                response = await self._client.post(
                    f"{BASE_URL}/{path}",
                    json=body,
                    headers=headers,
                    timeout=REQUEST_TIMEOUT_S,
                )
            except httpx.HTTPError as error:
                raise LanguageModelError(f"Muse request failed: {type(error).__name__}") from error
            if response.is_success:
                return response
            attempt += 1
            if response.status_code not in RETRYABLE or attempt == MAX_ATTEMPTS:
                raise LanguageModelError(
                    f"Muse returned {response.status_code}: {_error_message(response)}"
                )
            await asyncio.sleep(_wait_s(response, attempt))

    def _charges(self, usage: "_Usage") -> tuple[Charge, ...]:
        cached = usage.prompt_tokens_details.cached_tokens if usage.prompt_tokens_details else 0
        details = usage.completion_tokens_details
        reasoning = details.reasoning_tokens if details else 0
        return self._priced(usage.prompt_tokens, cached, usage.completion_tokens, reasoning)

    def _priced(
        self, prompt: int, cached: int, completion: int, reasoning: int
    ) -> tuple[Charge, ...]:
        # Reasoning is included in the output count; split it out so each token counts once.
        quantities = (
            ("input_token", prompt - cached, self._prices.input),
            ("cached_input_token", cached, self._prices.cached_input),
            ("output_token", completion - reasoning, self._prices.output),
            ("reasoning_token", reasoning, self._prices.output),
        )
        return tuple(
            Charge(
                provider=PROVIDER,
                model=self._model,
                unit=unit,
                quantity=quantity,
                usd=quantity * price / PER_MILLION,
            )
            for unit, quantity, price in quantities
        )


# Requests that share a leading system prompt route together, which raises prefix-cache hits.
def _prompt_cache_key(messages: tuple[ChatMessage, ...]) -> str | None:
    if not messages or not isinstance(messages[0], SystemMessage):
        return None
    return f"rewind-{hashlib.sha256(messages[0].text.encode()).hexdigest()[:16]}"


def _image_type(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise LanguageModelError("images must be JPEG, PNG or WebP")


def _wait_s(response: httpx.Response, attempt: int) -> float:
    fallback = float(2 ** (attempt - 1))
    header = response.headers.get("retry-after")
    if header is None:
        return fallback
    try:
        wait = float(header)
    except ValueError:
        try:
            wait = (parsedate_to_datetime(header) - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError):
            wait = fallback
    return min(max(wait, 0.0), MAX_WAIT_S)


def _error_message(response: httpx.Response) -> str:
    try:
        return _ErrorResponse.model_validate_json(response.content).error.message
    except ValidationError:
        return response.reason_phrase


class _Function(BaseModel):
    name: str
    arguments: str


class _ToolCall(BaseModel):
    id: str
    function: _Function


class _Message(BaseModel):
    content: str | None = None
    tool_calls: list[_ToolCall] | None = None


class _Choice(BaseModel):
    message: _Message


class _PromptDetails(BaseModel):
    cached_tokens: int = 0


class _CompletionDetails(BaseModel):
    reasoning_tokens: int = 0


class _Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    prompt_tokens_details: _PromptDetails | None = None
    completion_tokens_details: _CompletionDetails | None = None


class _ChatResponse(BaseModel):
    choices: list[_Choice] = Field(min_length=1)
    usage: _Usage


class _OutputText(BaseModel):
    type: str
    text: str = ""


class _OutputItem(BaseModel):
    type: str
    content: list[_OutputText] = []


class _InputDetails(BaseModel):
    cached_tokens: int = 0


class _OutputDetails(BaseModel):
    reasoning_tokens: int = 0


class _ResponseUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    input_tokens_details: _InputDetails | None = None
    output_tokens_details: _OutputDetails | None = None


class _Response(BaseModel):
    status: str
    output: list[_OutputItem] = []
    usage: _ResponseUsage


class _Error(BaseModel):
    message: str


class _ErrorResponse(BaseModel):
    error: _Error
