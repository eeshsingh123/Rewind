# Live smoke run of upload-and-ask, with real Muse and SAM calls; never run in CI.
#   uv run python scripts/smoke_ask.py data/sim/s1_jar_line-seed2/video.mp4
#       --ask "Anything wrong here?" --ask "Which jars have no label?"
#   uv run python scripts/smoke_ask.py path/to/photo.jpg --ask "Does each box have a label?"
# Prints the first look, each step as it happens, and each answer with its proof and cost.
# Reruns are free: every Muse look and SAM track is cached by its inputs.

import argparse
import asyncio
import logging
from decimal import Decimal
from pathlib import Path

import httpx

from rewind.adapters.storage.sql import SqlRepository
from rewind.app.ask import build_ask
from rewind.app.perception import build_perception
from rewind.app.settings import Settings
from rewind.core.analysis import clock_time
from rewind.core.ask import Step
from rewind.core.budget import EPOCH, BudgetScope


async def show(step: Step) -> None:
    print(f"  {'✓' if step.done else '…'} {step.text}", flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", type=Path)
    parser.add_argument("--ask", action="append", default=[])
    parser.add_argument("--data-root", type=Path, default=Path("data/smoke-ask"))
    parser.add_argument("--cap-usd", type=Decimal, default=Decimal("2.00"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = Settings(
        data_root=args.data_root, kill_switch_usd=args.cap_usd,
        workspace_daily_cap_usd=args.cap_usd,
    )  # fmt: skip
    args.data_root.mkdir(parents=True, exist_ok=True)
    repository = SqlRepository.from_url(settings.database_url_or_default)
    await repository.create_tables()
    async with httpx.AsyncClient(timeout=httpx.Timeout(300)) as client:
        ask = build_ask(settings, client, build_perception(settings, client, repository))
        blobs = ask.perception.blobs
        session_id = f"smoke-{args.clip.parent.name}-{args.clip.stem}"
        key = f"uploads/{session_id}/{args.clip.name}"
        await blobs.write_file(key, args.clip)
        scope = BudgetScope("smoke", session_id)
        model, segmenters = ask.model(scope), ask.segmenters(scope)

        session, _ = await ask.sessions.open(session_id, key, model, show)
        look = session.first_look
        print(f"\n{look.summary}", flush=True)
        for moment in look.moments:
            when = f"{clock_time(moment.span.start)} to {clock_time(moment.span.end)} "
            print(f"  {'' if session.photo else when}{moment.what}")
        print("Try:", *look.questions, sep="\n  ", flush=True)

        for question in args.ask:
            print(f"\n--- you: {question}", flush=True)
            answer = await ask.agent.ask(session, question, model, segmenters, show)
            session = await ask.sessions.get(session_id) or session
            for finding in answer.findings:
                print(f"- {finding.text} ({finding.kind})", flush=True)
                if finding.proof_key:
                    print(f"    ▶ {await blobs.local_path(finding.proof_key)}", flush=True)
            if not answer.findings:
                print(answer.text, flush=True)
            print(f"cost ${answer.cost_usd:.4f}", flush=True)
        print(f"\nledger: ${await repository.spent_usd(EPOCH)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
