import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from rewind.adapters.models.muse import MuseLanguageModel
from rewind.contracts import RuleSpec
from rewind.core.rules import RuleCompiler
from tests.fakes import FakeBlobStore, replay

EXAMPLES = json.loads(
    (Path(__file__).parent / "fixtures" / "rule_examples.json").read_text(encoding="utf-8")
)


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
