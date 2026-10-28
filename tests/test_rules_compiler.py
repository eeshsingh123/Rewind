import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from rewind.adapters.models.muse import MuseLanguageModel
from rewind.contracts import RuleSpec
from rewind.core.ports import AssistantMessage, LanguageModelError, UserMessage
from rewind.core.rules import RuleCompileError, RuleCompiler
from tests.fakes import FakeBlobStore, FakeLanguageModel, replay

EXAMPLES = json.loads(
    (Path(__file__).parent / "fixtures" / "rule_examples.json").read_text(encoding="utf-8")
)

VALID = json.dumps(
    {
        "entity": {"key": "jar", "phrase": "jar"},
        "parts": [{"key": "cap", "phrase": "cap"}],
        "zones": [],
        "checkpoint": "exits_view",
        "checkpoint_zone": None,
        "requirements": [
            {"kind": "has", "part": "cap", "zone": None, "n": None, "timing": "at_checkpoint"}
        ],
        "order": [],
        "dwell_zone": None,
        "dwell_max_s": None,
        "rate_min_per_minute": None,
        "unsupported": ["must be blue"],
    }
)
# The cap is used but never declared as a part, which RuleSpec rejects.
UNDECLARED_PART = VALID.replace('[{"key": "cap", "phrase": "cap"}]', "[]")


# Phrases and the wording of unsupported clauses are free text; the structure must match.
def shape(spec: RuleSpec) -> dict[str, Any]:
    dumped = spec.model_dump(mode="json")
    dumped["parts"] = sorted(dumped["parts"])
    dumped["entity"] = dumped["entity"]["key"]
    dumped["unsupported"] = len(dumped["unsupported"])
    return dumped


@pytest.mark.parametrize("example", EXAMPLES, ids=[e["sentence"] for e in EXAMPLES])
def test_example_sentences_compile_to_the_expected_spec(example: dict[str, Any]) -> None:
    sent: list[httpx.Request] = []

    async def compile_recorded() -> Any:
        async with replay("muse/compile_examples.json", sent) as client:
            model = MuseLanguageModel(
                client, "test-key", "muse-spark-1.3-contributor", FakeBlobStore()
            )
            return await RuleCompiler(model, FakeBlobStore()).compile(example["sentence"])

    compiled = asyncio.run(compile_recorded())

    assert shape(compiled.spec) == shape(RuleSpec.model_validate(example["spec"]))
    assert sorted(compiled.zone_phrases) == example["zones"]
    assert [charge.unit for charge in compiled.charges] == [
        "input_token",
        "cached_input_token",
        "output_token",
        "reasoning_token",
    ]


def test_an_invalid_reply_is_retried_with_the_error_fed_back() -> None:
    model = FakeLanguageModel([UNDECLARED_PART, VALID])

    compiled = asyncio.run(RuleCompiler(model, FakeBlobStore()).compile("Each jar needs a cap."))

    retry = model.requests[1].messages
    assert retry[-2] == AssistantMessage(text=UNDECLARED_PART)
    assert isinstance(retry[-1], UserMessage)
    assert "parts used but not declared: cap" in retry[-1].text
    assert compiled.spec.unsupported == ("must be blue",)
    assert len(compiled.charges) == 2
    assert model.requests[0].effort == "minimal"
    assert model.requests[0].response_schema is not None


def test_compiling_gives_up_after_bounded_retries_and_keeps_the_charges() -> None:
    model = FakeLanguageModel([UNDECLARED_PART] * 3)

    with pytest.raises(RuleCompileError, match="after 3 attempts") as failure:
        asyncio.run(RuleCompiler(model, FakeBlobStore()).compile("Each jar needs a cap."))

    assert len(failure.value.charges) == 3


def test_a_model_error_keeps_the_charges_of_earlier_attempts() -> None:
    model = FakeLanguageModel([UNDECLARED_PART, LanguageModelError("Muse returned 400")])

    with pytest.raises(RuleCompileError, match="400") as failure:
        asyncio.run(RuleCompiler(model, FakeBlobStore()).compile("Each jar needs a cap."))

    assert len(failure.value.charges) == 1


def test_compiled_rules_are_cached_by_normalized_text_and_model_version() -> None:
    cache = FakeBlobStore()
    model = FakeLanguageModel([VALID])
    other_model = FakeLanguageModel([VALID], model_version="fake-2")

    async def compile_three() -> list[Any]:
        first = await RuleCompiler(model, cache).compile("Each jar needs a cap.")
        again = await RuleCompiler(model, cache).compile("  each JAR needs a cap ")
        other = await RuleCompiler(other_model, cache).compile("Each jar needs a cap.")
        return [first, again, other]

    first, again, other = asyncio.run(compile_three())

    assert again.spec == first.spec
    assert again.charges == ()
    assert len(model.requests) == 1
    assert len(other_model.requests) == 1
    assert other.charges
    assert len(cache.blobs) == 2
