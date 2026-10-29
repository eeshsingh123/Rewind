from datetime import UTC, datetime
from decimal import Decimal

import pytest

from rewind.contracts import Charge, CostEntry
from rewind.core.budget import (
    BudgetedSegmenter,
    BudgetExceededError,
    BudgetLimits,
    BudgetScope,
    Ledger,
)
from rewind.core.ports import ConceptPrompt, Segmentation, SegmentRequest

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 11, 2, 15, 0, tzinfo=UTC)


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


def _charge(usd: str) -> Charge:
    return Charge(
        provider="meta", model="sam-3.1", unit="video_frame", quantity=50, usd=Decimal(usd)
    )


def _spent(usd: str, workspace_id: str, at: datetime = NOW) -> CostEntry:
    return CostEntry(
        **_charge(usd).model_dump(), id=f"e-{usd}-{workspace_id}", at=at, workspace_id=workspace_id
    )


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
        return Segmentation(observations=(), charges=(_charge("0.0100"),))


async def test_a_call_over_the_workspace_daily_cap_is_refused_before_it_is_made() -> None:
    yesterday = NOW.replace(day=1)
    costs = MemoryCosts(
        _spent("0.995", "w1"), _spent("5", "w1", at=yesterday), _spent("3", "other")
    )
    limits = BudgetLimits(workspace_daily_usd=Decimal("1.00"), kill_switch_usd=Decimal(120))
    inner = FakeSegmenter()
    request = SegmentRequest(
        footage_id="f1", media_key="proxy.mp4", width=1280, height=720, frame_times=(0.0,),
        prompt=ConceptPrompt("carton"),
    )  # fmt: skip
    ledger = Ledger(costs, limits, clock=lambda: NOW)
    segmenter = BudgetedSegmenter(inner, ledger, BudgetScope(workspace_id="w1", job_id="j1"))

    with pytest.raises(BudgetExceededError, match="daily cap"):
        await segmenter.segment(request)
    assert inner.calls == 0
