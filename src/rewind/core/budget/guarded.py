from dataclasses import dataclass
from decimal import Decimal

from rewind.contracts import Charge, CostEstimate
from rewind.core.budget.ledger import Ledger
from rewind.core.ports import (
    Completion,
    CompletionRequest,
    LanguageModel,
    Segmentation,
    Segmenter,
    SegmentRequest,
    UserMessage,
)

# Rough sizes for a pre-call estimate; the ledger records the provider's real counts after.
_CHARS_PER_TOKEN = 4
_TOKENS_PER_IMAGE = 1600


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
    def min_billed_frames(self) -> int:
        return self._inner.min_billed_frames

    def estimate(self, request: SegmentRequest) -> tuple[Charge, ...]:
        return self._inner.estimate(request)

    async def segment(self, request: SegmentRequest) -> Segmentation:
        estimate = CostEstimate(charges=self._inner.estimate(request)).total_usd
        async with self._ledger.reserve(self._scope.workspace_id, estimate):
            result = await self._inner.segment(request)
            await self._ledger.record(self._scope.workspace_id, self._scope.job_id, result.charges)
        return result


@dataclass(frozen=True, slots=True)
class TokenPrices:
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
            completion = await self._inner.complete(request)
            await self._ledger.record(
                self._scope.workspace_id, self._scope.job_id, completion.charges
            )
        return completion


# The most a call can cost: every input token uncached and the whole output budget used.
def estimate_completion_usd(request: CompletionRequest, prices: TokenPrices) -> Decimal:
    chars = sum(len(message.text) for message in request.messages)
    chars += sum(len(tool.description) + len(str(tool.parameters)) for tool in request.tools)
    chars += len(str(request.response_schema or ""))
    images = sum(
        len(message.image_keys) for message in request.messages if isinstance(message, UserMessage)
    )
    input_tokens = chars // _CHARS_PER_TOKEN + images * _TOKENS_PER_IMAGE
    million = Decimal(1_000_000)
    return (
        input_tokens * prices.input_usd_per_million
        + request.max_output_tokens * prices.output_usd_per_million
    ) / million
