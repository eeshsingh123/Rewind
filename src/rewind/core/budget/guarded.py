import math
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from rewind.contracts import Charge, CostEstimate
from rewind.core.budget.ledger import Ledger
from rewind.core.ports import (
    SPEECH_BYTES_PER_S,
    Completion,
    CompletionRequest,
    LanguageModel,
    Segmentation,
    Segmenter,
    SegmentRequest,
    Transcriber,
    Transcript,
    UserMessage,
)

# Rough sizes for a pre-call estimate; the ledger records the provider's real counts after.
_CHARS_PER_TOKEN = 4
_TOKENS_PER_IMAGE = 1600
# The 60 s jar clip was 15.7k input tokens; this covers a few minutes of video.
_TOKENS_PER_VIDEO = 100_000


@dataclass(frozen=True, slots=True)
class BudgetScope:
    workspace_id: str
    job_id: str | None = None


class BudgetedSegmenter:
    def __init__(self, inner: Segmenter, ledger: Ledger, scope: BudgetScope) -> None:
        self._inner = inner
        self._ledger = ledger
        self._scope = scope

    @property
    def model_version(self) -> str:
        return self._inner.model_version

    @property
    def max_frames(self) -> int:
        return self._inner.max_frames

    @property
    def max_prompt_objects(self) -> int:
        return self._inner.max_prompt_objects

    @property
    def max_tracked_objects(self) -> int | None:
        return self._inner.max_tracked_objects

    @property
    def min_billed_frames(self) -> int:
        return self._inner.min_billed_frames

    def estimate(self, request: SegmentRequest) -> tuple[Charge, ...]:
        return self._inner.estimate(request)

    async def segment(self, request: SegmentRequest) -> Segmentation:
        estimate = CostEstimate(charges=self._inner.estimate(request)).total_usd
        async with self._ledger.reserve(self._scope.workspace_id, estimate):
            try:
                result = await self._inner.segment(request)
            except Exception:
                await self._ledger.record_failure(
                    self._scope.workspace_id, self._scope.job_id, self._inner.estimate(request)
                )
                raise
            await self._ledger.record(self._scope.workspace_id, self._scope.job_id, result.charges)
        return result


class BudgetedTranscriber:
    def __init__(self, inner: Transcriber, ledger: Ledger, scope: BudgetScope) -> None:
        self._inner = inner
        self._ledger = ledger
        self._scope = scope

    @property
    def model_version(self) -> str:
        return self._inner.model_version

    @property
    def max_seconds(self) -> int:
        return self._inner.max_seconds

    def estimate(self, seconds: int) -> tuple[Charge, ...]:
        return self._inner.estimate(seconds)

    async def transcribe(self, source: Path) -> Transcript:
        charges = self.estimate(math.ceil(source.stat().st_size / SPEECH_BYTES_PER_S))
        estimate = CostEstimate(charges=charges).total_usd
        async with self._ledger.reserve(self._scope.workspace_id, estimate):
            try:
                transcript = await self._inner.transcribe(source)
            except Exception:
                await self._ledger.record_failure(
                    self._scope.workspace_id, self._scope.job_id, charges
                )
                raise
            await self._ledger.record(
                self._scope.workspace_id, self._scope.job_id, transcript.charges
            )
        return transcript


@dataclass(frozen=True, slots=True)
class TokenPrices:
    provider: str
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal


class BudgetedLanguageModel:
    def __init__(
        self, inner: LanguageModel, ledger: Ledger, prices: TokenPrices, scope: BudgetScope
    ) -> None:
        self._inner = inner
        self._ledger = ledger
        self._prices = prices
        self._scope = scope

    @property
    def model_version(self) -> str:
        return self._inner.model_version

    async def complete(self, request: CompletionRequest) -> Completion:
        estimate = estimate_completion_usd(request, self._prices)
        async with self._ledger.reserve(self._scope.workspace_id, estimate):
            try:
                completion = await self._inner.complete(request)
            except Exception:
                charge = Charge(
                    provider=self._prices.provider,
                    model=self._inner.model_version,
                    unit="request",
                    quantity=1,
                    usd=estimate,
                )
                await self._ledger.record_failure(
                    self._scope.workspace_id, self._scope.job_id, (charge,)
                )
                raise
            await self._ledger.record(
                self._scope.workspace_id, self._scope.job_id, completion.charges
            )
        return completion


# The most a call can cost: every input token uncached and the whole output budget used.
def estimate_completion_usd(request: CompletionRequest, prices: TokenPrices) -> Decimal:
    chars = sum(len(message.text) for message in request.messages)
    chars += sum(len(tool.description) + len(str(tool.parameters)) for tool in request.tools)
    chars += len(str(request.response_schema or ""))
    users = [message for message in request.messages if isinstance(message, UserMessage)]
    images = sum(len(message.image_keys) for message in users)
    videos = sum(message.video_key is not None for message in users)
    input_tokens = (
        chars // _CHARS_PER_TOKEN + images * _TOKENS_PER_IMAGE + videos * _TOKENS_PER_VIDEO
    )
    million = Decimal(1_000_000)
    return (
        input_tokens * prices.input_usd_per_million
        + request.max_output_tokens * prices.output_usd_per_million
    ) / million
