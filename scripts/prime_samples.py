# Runs every sample once with real Muse and SAM calls, so later taps on the page are instant and
# free; never run in CI. Asks each question from its first look. Reruns cost nothing.
#   uv run python scripts/prime_samples.py [--only jars] [--cap-usd 3]

import argparse
import asyncio
import logging
from decimal import Decimal

import httpx

from rewind.adapters.storage.sql import SqlRepository
from rewind.app.ask import build_ask
from rewind.app.perception import build_perception
from rewind.app.settings import Settings
from rewind.core.ask import Step
from rewind.core.budget import BudgetScope


async def show(step: Step) -> None:
    if step.done:
        print(f"  {step.text}", flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="One sample's name; the default is all of them.")
    parser.add_argument("--cap-usd", type=Decimal, default=Decimal("3.00"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = Settings(workspace_daily_cap_usd=args.cap_usd)
    settings.data_root.mkdir(parents=True, exist_ok=True)
    repository = SqlRepository.from_url(settings.database_url_or_default)
    await repository.create_tables()
    async with httpx.AsyncClient(timeout=httpx.Timeout(300)) as client:
        ask = build_ask(settings, client, build_perception(settings, client, repository))
        total = Decimal(0)
        for sample in ask.samples.available():
            if args.only and sample.name != args.only:
                continue
            print(f"\n{sample.title}", flush=True)
            session_id = f"prime-{sample.name}"
            scope = BudgetScope("prime", session_id)
            model, segmenters = ask.model(scope), ask.segmenters(scope)
            key = await ask.samples.stored(sample)
            session, charges = await ask.sessions.open(session_id, key, model, show)
            total += sum((charge.usd for charge in charges), Decimal(0))
            # Each from the same fresh session: a suggested question stands alone on the page.
            for question in session.first_look.questions:
                print(f"- {question}", flush=True)
                exchange = await ask.agent.ask(session, question, model, segmenters, show)
                total += exchange.cost_usd
                print(f"  {exchange.answer}\n  cost ${exchange.cost_usd:.4f}", flush=True)
        print(f"\nspent ${total:.4f}", flush=True)
        await repository.close()


if __name__ == "__main__":
    asyncio.run(main())
