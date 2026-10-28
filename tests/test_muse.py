import asyncio
import json
from decimal import Decimal

import httpx
import pytest

from rewind.adapters.models.muse import MuseLanguageModel
from rewind.contracts import Charge
from rewind.core.ports import Completion, CompletionRequest, Tool, ToolCall, UserMessage
from tests.fakes import FakeBlobStore, replay

MODEL = "muse-spark-1.3-contributor"
VIOLATIONS = Tool(
    name="violations",
    description="List rule violations for a line in a time window.",
    parameters={
        "type": "object",
        "additionalProperties": False,
        "required": ["line_id", "since_minutes"],
        "properties": {"line_id": {"type": "string"}, "since_minutes": {"type": "integer"}},
    },
)


def complete(
    recording: str, request: CompletionRequest, match: bool
) -> tuple[Completion, list[httpx.Request]]:
    sent: list[httpx.Request] = []

    async def run() -> Completion:
        async with replay(recording, sent, match) as client:
            return await MuseLanguageModel(client, "test-key", MODEL, FakeBlobStore()).complete(
                request
            )

    return asyncio.run(run()), sent


def usd(per_million: str, quantity: int) -> Decimal:
    return Decimal(per_million) * quantity / 1_000_000


def test_a_tool_call_reply_maps_to_tool_calls_and_one_charge_per_unit() -> None:
    request = CompletionRequest(
        messages=(UserMessage("Any problems on line L1 in the last hour?"),),
        max_output_tokens=1024,
        tools=(VIOLATIONS,),
    )

    completion, sent = complete("muse/tool_call.json", request, match=False)

    assert completion.text == "I'll check L1 for problems in the last hour."
    assert completion.tool_calls == (
        ToolCall(
            id="call_e090100c-591c-4fef-be86-a41933fa7c69",
            name="violations",
            arguments='{"line_id":"L1","since_minutes":60}',
        ),
    )
    # 568 prompt tokens with 497 cached; 222 completion tokens of which 134 are reasoning.
    assert completion.charges == tuple(
        Charge(provider="meta", model=MODEL, unit=unit, quantity=quantity, usd=usd(price, quantity))
        for unit, quantity, price in [
            ("input_token", 71, "0.10"),
            ("cached_input_token", 497, "0.002"),
            ("output_token", 88, "0.20"),
            ("reasoning_token", 134, "0.20"),
        ]
    )
    body = json.loads(sent[0].content)
    assert sent[0].headers["authorization"] == "Bearer test-key"
    assert body["tools"][0]["function"]["name"] == "violations"
    assert body["reasoning_effort"] == "minimal"
    assert "prompt_cache_key" not in body


def test_a_server_error_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []

    async def no_wait(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", no_wait)
    request = CompletionRequest(
        messages=(UserMessage("every carton needs a label when it enters sealing"),),
        max_output_tokens=2048,
    )

    completion, sent = complete("muse/server_error_then_success.json", request, match=False)

    assert len(sent) == 2
    assert waits == [1.0]
    assert '"checkpoint": "enters"' in completion.text
