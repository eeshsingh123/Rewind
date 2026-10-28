# Live smoke for T3: compiles the example sentences against real Muse Spark. Not run in CI.
# uv run python scripts/smoke_rules.py [--record tests/recordings/muse/compile_examples.json]
import argparse
import asyncio
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from rewind.adapters.models import muse
from rewind.adapters.models.muse import MuseLanguageModel
from rewind.app.settings import Settings
from rewind.contracts import RuleSpec
from rewind.core.rules import RuleCompiler

EXAMPLES = Path(__file__).parents[1] / "tests" / "fixtures" / "rule_examples.json"
KEPT_HEADERS = {"content-type", "date", "x-request-id"}


class MemoryBlobStore:
    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    async def write(self, key: str, data: bytes) -> None:
        self.blobs[key] = data

    async def write_file(self, key: str, source: Path) -> None:
        self.blobs[key] = source.read_bytes()

    async def read(self, key: str) -> bytes:
        return self.blobs[key]

    async def local_path(self, key: str) -> Path:
        raise NotImplementedError

    async def exists(self, key: str) -> bool:
        return key in self.blobs

    def signed_url(self, key: str, expires_in_s: int) -> str:
        raise NotImplementedError


def shape(spec: RuleSpec) -> dict[str, Any]:
    # Phrases and the wording of unsupported clauses are free text; the structure must match.
    dumped = spec.model_dump(mode="json")
    dumped["parts"] = sorted(dumped["parts"])
    dumped["entity"] = dumped["entity"]["key"]
    dumped["unsupported"] = len(dumped["unsupported"])
    return dumped


async def main(record: Path | None, cache_key: bool) -> None:
    settings = Settings()
    if settings.meta_api_key is None:
        raise SystemExit("META_API_KEY is not set")
    if not cache_key:
        muse._prompt_cache_key = lambda messages: None  # type: ignore[assignment]
    exchanges: list[dict[str, Any]] = []

    async def keep(response: httpx.Response) -> None:
        await response.aread()
        exchanges.append(
            {
                "request": {
                    "method": response.request.method,
                    "url": str(response.request.url),
                    "json": json.loads(response.request.content),
                },
                "status": response.status_code,
                "response_headers": {
                    name: value
                    for name, value in response.headers.items()
                    if name in KEPT_HEADERS or name.startswith("x-ratelimit-")
                },
                "elapsed_s": round(response.elapsed.total_seconds(), 2),
                "body": response.json(),
            }
        )

    examples = json.loads(EXAMPLES.read_text(encoding="utf-8"))
    total = Decimal(0)
    matches = 0
    async with httpx.AsyncClient(event_hooks={"response": [keep]}) as client:
        model = MuseLanguageModel(
            client, settings.meta_api_key.get_secret_value(), settings.muse_model, MemoryBlobStore()
        )
        compiler = RuleCompiler(model, MemoryBlobStore())
        for example in examples:
            calls_before = len(exchanges)
            compiled = await compiler.compile(example["sentence"])
            expected = RuleSpec.model_validate(example["spec"])
            cost = sum((charge.usd for charge in compiled.charges), Decimal(0))
            total += cost
            calls = exchanges[calls_before:]
            usage = [call["body"].get("usage", call["status"]) for call in calls]
            zones = sorted(compiled.zone_phrases)
            same = shape(compiled.spec) == shape(expected) and zones == sorted(example["zones"])
            matches += same
            print(f"\n{'MATCH' if same else 'DIFF '} {example['sentence']}")
            print(f"  attempts {len(usage)}, cost ${cost:.6f}, usage {usage}")
            print(f"  got      {compiled.spec.model_dump_json(exclude_none=True)}")
            print(f"  zones    {dict(compiled.zone_phrases)}")
            if not same:
                print(f"  expected {expected.model_dump_json(exclude_none=True)}")
    print(f"\n{matches}/{len(examples)} match; total ${total:.6f} on {settings.muse_model}")
    if record is not None:
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(exchanges, indent=2) + "\n", encoding="utf-8")
        print(f"recorded {len(exchanges)} exchanges to {record}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path)
    parser.add_argument("--no-cache-key", action="store_true")
    arguments = parser.parse_args()
    asyncio.run(main(arguments.record, not arguments.no_cache_key))
