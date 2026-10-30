# Live smoke run of a check and its insights, with real SAM and Muse calls; never run in CI.
#   uv run python scripts/smoke_insights.py data/sim/s1_jar_line-seed2/video.mp4
#       --rule "Every jar has a label when it leaves the view." --ask "when did it stop?"
# Prints what the chat would get: the checklist, the result, each insight clip's size, and the
# answer to each --ask. Reruns are free: perception, rules and insights are cached.

import argparse
import asyncio
import hashlib
import logging
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import httpx

from rewind.adapters.models.muse import PRICES, MuseLanguageModel
from rewind.adapters.storage.sql import SqlRepository
from rewind.adapters.video.contact_sheet import OpenCvContactSheet
from rewind.adapters.video.proof_clip import OpenCvProofClips
from rewind.app.perception import build_perception
from rewind.app.settings import Settings
from rewind.contracts import Button, ChannelCapabilities, Line, ReplyMedia, Rule
from rewind.core.analysis import Analysis, AnalysisSettings, Insights, InsightWriter
from rewind.core.budget import EPOCH, BudgetedLanguageModel, BudgetScope, TokenPrices
from rewind.core.ports import LanguageModel
from rewind.core.rules import RuleCompiler, normalize

CHAT_MAX_BYTES = 16 * 1024 * 1024
CAPABILITIES = ChannelCapabilities(
    max_text_chars=4096, max_media_bytes=CHAT_MAX_BYTES, max_buttons=8,
    max_image_bytes=CHAT_MAX_BYTES, can_edit=True,
)  # fmt: skip


class PrintingNotifier:
    async def notify(
        self, channel: str, chat_id: str, text: str,
        media: tuple[ReplyMedia, ...] = (), buttons: tuple[Button, ...] = (),
    ) -> None:  # fmt: skip
        body = text or "\n".join(m.caption or "" for m in media)
        print(f"\n--- message ---\n{body}\n[{' | '.join(b.label for b in buttons)}]", flush=True)
        self.buttons = buttons

    async def progress(
        self, channel: str, chat_id: str, key: str, text: str, buttons: tuple[Button, ...] = ()
    ) -> None:
        self.last_progress = text


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", type=Path)
    parser.add_argument("--rule", action="append", required=True)
    parser.add_argument("--ask", action="append", default=[])
    parser.add_argument("--data-root", type=Path, default=Path("data/smoke-insights"))
    parser.add_argument("--cap-usd", type=Decimal, default=Decimal("1.00"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = Settings(
        data_root=args.data_root, kill_switch_usd=args.cap_usd,
        workspace_daily_cap_usd=args.cap_usd,
    )  # fmt: skip
    if settings.meta_api_key is None:
        raise SystemExit("META_API_KEY is not set")
    args.data_root.mkdir(parents=True, exist_ok=True)
    repository = SqlRepository.from_url(settings.database_url_or_default)
    await repository.create_tables()
    async with httpx.AsyncClient(timeout=httpx.Timeout(300)) as client:
        perception = build_perception(settings, client, repository)
        blobs = perception.blobs
        muse = MuseLanguageModel(
            client, settings.meta_api_key.get_secret_value(), settings.muse_model, blobs
        )
        price = PRICES[settings.muse_model]

        def model_for(workspace_id: str) -> LanguageModel:
            return BudgetedLanguageModel(
                muse, perception.ledger, TokenPrices("meta", price.input, price.output),
                BudgetScope(workspace_id),
            )  # fmt: skip

        notifier = PrintingNotifier()
        analysis = Analysis(
            repository=repository, blobs=blobs, ingest=perception.ingest,
            planner=perception.planner, runner=perception.runner,
            observations=perception.observations, segmenters=perception,
            compiler_for=lambda ws: RuleCompiler(model_for(ws), blobs),
            video=perception.video_tools, stills=perception.video_tools,
            clips=OpenCvProofClips(),
            insights=Insights(repository, blobs, OpenCvContactSheet(),
                              lambda ws: InsightWriter(model_for(ws), blobs)),
            notifier=notifier, capabilities={"smoke": CAPABILITIES},
            settings=AnalysisSettings(fps=settings.track_fps),
        )  # fmt: skip
        now = datetime.now(UTC)
        # One line per clip, and one id per rule text, so clips never count as each other's history.
        line_id = f"smoke-line-{args.clip.parent.name}"
        line = Line(id=line_id, workspace_id="smoke", name="Smoke line", created_at=now)
        await repository.save_line(line)
        rule_ids: list[str] = []
        for text in args.rule:
            compiled = await RuleCompiler(model_for("smoke"), blobs).compile(text)
            rule = Rule(
                id=f"smoke-rule-{hashlib.sha256(normalize(text).encode()).hexdigest()[:12]}",
                line_id=line.id, text=text,
                normalized_text=normalize(text), spec=compiled.spec, created_at=now,
            )  # fmt: skip
            await repository.save_rule(rule)
            rule_ids.append(rule.id)
        footage_id = f"smoke-{args.clip.stem}-{args.clip.parent.name}"
        key = f"footage/{footage_id}/{args.clip.name}"
        await blobs.write_file(key, args.clip)
        base = {"workspace_id": "smoke", "line_id": line.id, "channel": "smoke", "chat_id": "c"}
        read = await analysis.read({**base, "footage_id": footage_id, "original_key": key})
        assert read is not None
        print(read.summary, flush=True)
        job = {"footage_ids": [footage_id], "rule_ids": rule_ids, "job_id": f"job-{now:%H%M%S}"}
        await analysis.handle_analyze({**base, **job})
        print(f"\n--- checklist ---\n{notifier.last_progress}", flush=True)
        footage = await repository.get_footage(footage_id)
        assert footage is not None
        for button in notifier.buttons:
            if not button.value.startswith("/insight "):
                continue
            insight = await analysis.insight(footage, button.value.split()[-1])
            assert insight is not None
            clip = await analysis.insight_clip(footage, insight, CHAT_MAX_BYTES)
            for media in clip.media:
                path = await blobs.local_path(media.blob_key)
                print(f"{button.label}: {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
            print("  rests on:", *insight.facts, sep="\n    ")
        for question in args.ask:
            answer = await analysis.ask("smoke", footage, question, None)
            print(f"\n--- you: {question}\n{answer.text if answer else '(not checked)'}",
                  flush=True)  # fmt: skip
            if answer is not None and answer.insight is not None:
                print(f"[▶ {answer.insight.label}]", flush=True)
        print(f"\nledger: ${await repository.spent_usd(EPOCH)}", flush=True)
        await repository.close()


if __name__ == "__main__":
    asyncio.run(main())
