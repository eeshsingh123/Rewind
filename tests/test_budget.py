from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from rewind.app.settings import Settings
from rewind.contracts import Charge, CostEntry
from rewind.core.budget import (
    BudgetedLanguageModel,
    BudgetedSegmenter,
    BudgetExceededError,
    BudgetLimits,
    BudgetScope,
    Ledger,
    TokenPrices,
)
from rewind.core.ports import (
    Completion,
    CompletionRequest,
    ConceptPrompt,
    Segmentation,
    SegmenterError,
    SegmentRequest,
    UserMessage,
)

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 11, 2, 15, 0, tzinfo=UTC)
SCOPE = BudgetScope(workspace_id="w1", job_id="j1")


class MemoryCosts:
    def __init__(self, *entries: CostEntry) -> None:
        self.entries = list(entries)

    async def add_cost(self, entry: CostEntry) -> None:
        self.entries.append(entry)

    async def spent_usd(self, since: datetime, workspace_id: str | None = None) -> Decimal:
        return sum(
            (
                e.usd
                for e in self.entries
                if e.at >= since and workspace_id in (None, e.workspace_id)
            ),
            Decimal(0),
        )


def _charge(usd: str, quantity: int = 50) -> Charge:
    return Charge(
        provider="meta", model="sam-3.1", unit="video_frame", quantity=quantity, usd=Decimal(usd)
    )


def _spent(usd: str, workspace_id: str, at: datetime = NOW) -> CostEntry:
    return CostEntry(
        **_charge(usd).model_dump(), id=f"e-{usd}-{workspace_id}", at=at, workspace_id=workspace_id
    )


# Bills 50 frames per call and reports slightly more than estimated, as a provider might.
class FakeSegmenter:
    model_version = "sam-3.1"
    max_frames = 15_000
    max_prompt_objects = 16
    max_tracked_objects = 16
    min_billed_frames = 50

    def __init__(self) -> None:
        self.calls = 0

    def estimate(self, request: SegmentRequest) -> tuple[Charge, ...]:
        return (_charge("0.0100"),)

    async def segment(self, request: SegmentRequest) -> Segmentation:
        self.calls += 1
        return Segmentation(observations=(), charges=(_charge("0.0120", 60),))


REQUEST = SegmentRequest(
    footage_id="f1",
    media_key="proxy.mp4",
    width=1280,
    height=720,
    frame_times=(0.0,),
    prompt=ConceptPrompt("carton"),
)


def _ledger(costs: MemoryCosts, daily: str = "1.00", kill: str = "120") -> Ledger:
    limits = BudgetLimits(workspace_daily_usd=Decimal(daily), kill_switch_usd=Decimal(kill))
    return Ledger(costs, limits, clock=lambda: NOW)


async def test_a_paid_call_records_the_reported_charges_not_the_estimate() -> None:
    costs = MemoryCosts()
    inner = FakeSegmenter()

    await BudgetedSegmenter(inner, _ledger(costs), SCOPE).segment(REQUEST)

    assert [(e.quantity, e.usd, e.workspace_id, e.job_id, e.at) for e in costs.entries] == [
        (60, Decimal("0.0120"), "w1", "j1", NOW)
    ]


async def test_a_call_over_the_workspace_daily_cap_is_refused_before_it_is_made() -> None:
    yesterday = NOW.replace(day=1)
    costs = MemoryCosts(
        _spent("0.995", "w1"), _spent("5", "w1", at=yesterday), _spent("3", "other")
    )
    inner = FakeSegmenter()

    with pytest.raises(BudgetExceededError, match="daily cap"):
        await BudgetedSegmenter(inner, _ledger(costs), SCOPE).segment(REQUEST)
    assert inner.calls == 0


async def test_the_kill_switch_counts_every_workspace_and_every_day() -> None:
    costs = MemoryCosts(_spent("60", "a", at=NOW.replace(year=2025)), _spent("59.995", "b"))
    inner = FakeSegmenter()

    with pytest.raises(BudgetExceededError, match="kill switch"):
        await BudgetedSegmenter(inner, _ledger(costs, daily="100"), SCOPE).segment(REQUEST)
    assert inner.calls == 0


async def test_calls_in_flight_count_against_the_cap() -> None:
    ledger = _ledger(MemoryCosts(), daily="0.015")

    async with ledger.reserve("w1", Decimal("0.0100")):
        with pytest.raises(BudgetExceededError):
            async with ledger.reserve("w1", Decimal("0.0100")):
                pass


class FakeLanguageModel:
    model_version = "muse-spark-1.3-contributor"

    async def complete(self, request: CompletionRequest) -> Completion:
        charge = Charge(
            provider="meta",
            model=self.model_version,
            unit="output_token",
            quantity=40,
            usd=Decimal("0.000008"),
        )
        return Completion(text="{}", tool_calls=(), charges=(charge,))


async def test_the_language_model_is_guarded_by_its_worst_case_estimate() -> None:
    prices = TokenPrices(
        provider="meta",
        input_usd_per_million=Decimal("0.10"),
        output_usd_per_million=Decimal("0.20"),
    )
    costs = MemoryCosts()
    model = BudgetedLanguageModel(FakeLanguageModel(), _ledger(costs, daily="0.001"), prices, SCOPE)
    small = CompletionRequest(
        messages=(UserMessage(text="a jar needs a cap"),), max_output_tokens=500
    )
    huge = CompletionRequest(messages=(UserMessage(text="x"),), max_output_tokens=10_000)

    await model.complete(small)
    with pytest.raises(BudgetExceededError):
        await model.complete(huge)
    assert [e.unit for e in costs.entries] == ["output_token"]


class FailingSegmenter(FakeSegmenter):
    async def segment(self, request: SegmentRequest) -> Segmentation:
        raise SegmenterError("the stream ended before response.completed")


async def test_a_failed_call_is_recorded_as_unconfirmed_and_counts_toward_the_caps() -> None:
    costs = MemoryCosts()
    ledger = _ledger(costs, daily="0.015")

    with pytest.raises(SegmenterError):
        await BudgetedSegmenter(FailingSegmenter(), ledger, SCOPE).segment(REQUEST)

    assert [(e.unit, e.quantity, e.usd) for e in costs.entries] == [
        ("video_frame_unconfirmed", 50, Decimal("0.0100"))
    ]
    with pytest.raises(BudgetExceededError):
        await BudgetedSegmenter(FakeSegmenter(), ledger, SCOPE).segment(REQUEST)


def test_the_kill_switch_cannot_be_set_past_the_hard_limit() -> None:
    with pytest.raises(ValidationError):
        Settings(kill_switch_usd=Decimal("141"))
