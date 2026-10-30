# Live smoke run of perception against the real SAM 3.1 API; never run in CI. Usage:
#   uv run python scripts/smoke_perception.py CLIP --concept "rectangle" [--box T,X0,Y0,X1,Y1]
# Ingests the clip, tracks each concept on the busy parts twice (the second run must cost $0), and
# compares the ledger with the usage the provider reported. The budget guard caps the spend.

import argparse
import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import httpx

from rewind.adapters.storage.sql import SqlRepository
from rewind.app.perception import build_perception
from rewind.app.settings import Settings
from rewind.contracts import Box, SamplingPlan, SamplingWindow
from rewind.core.budget import EPOCH, BudgetScope
from rewind.core.ports import ObjectPrompt, ObjectPrompts, SegmentRequest

SAFE_HEADERS = ("content-type", "date", "x-request-id")


# Saves each response (never request headers) and the provider's reported usage.
class RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self, directory: Path | None) -> None:
        self._inner = httpx.AsyncHTTPTransport()
        self._directory = directory
        self.reported_frames = 0
        self._count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        body = await response.aread()
        text = body.decode(errors="replace")
        self._count += 1
        events = [line[6:] for line in text.splitlines() if line.startswith("data: ")]
        for event in events:
            if event != "[DONE]" and '"response.completed"' in event:
                usage = json.loads(event)["response"].get("usage") or {}
                self.reported_frames += usage.get("video_frames_processed", 0)
        if self._directory is not None:
            self._save(request, response, text, events)
        return httpx.Response(
            response.status_code, headers=response.headers, content=body, request=request
        )

    def _save(
        self, request: httpx.Request, response: httpx.Response, text: str, events: list[str]
    ) -> None:
        assert self._directory is not None
        self._directory.mkdir(parents=True, exist_ok=True)
        is_json = request.headers.get("content-type", "").startswith("application/json")
        record = {
            "request": {
                "method": request.method,
                "url": str(request.url),
                "json": json.loads(request.content) if is_json else None,
            },
            "status": response.status_code,
            "response_headers": {
                k: v for k, v in response.headers.items() if k.lower() in SAFE_HEADERS
            },
            "body": {"sse_events": [e if e == "[DONE]" else json.loads(e) for e in events]}
            if events
            else json.loads(text),
        }
        name = f"{self._count:02d}_{request.url.path.strip('/').replace('/', '_')}.json"
        (self._directory / name).write_text(json.dumps(record, indent=2), encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", type=Path)
    parser.add_argument("--concept", action="append", required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--box", help="t,x0,y0,x1,y1: also track one box forward from t")
    parser.add_argument("--data-root", type=Path, default=Path("data/smoke"))
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--cap-usd", type=Decimal, default=Decimal("0.50"))
    args = parser.parse_args()

    settings = Settings(
        data_root=args.data_root,
        track_fps=args.fps,
        kill_switch_usd=args.cap_usd,
        workspace_daily_cap_usd=args.cap_usd,
    )
    transport = RecordingTransport(args.record_dir)
    timeout = httpx.Timeout(600, connect=30)
    repository = SqlRepository.from_url(settings.database_url_or_default)
    await repository.create_tables()
    async with httpx.AsyncClient(transport=transport, timeout=timeout) as client:
        perception = build_perception(settings, client, repository)
        original_key = f"footage/{args.clip.name}"
        await perception.blobs.write_file(original_key, args.clip)
        ingested = await perception.ingest.ingest("smoke", "smoke", original_key)
        footage = ingested.footage
        print(
            f"footage {footage.duration_s:.1f}s @ {footage.fps:g} fps, "
            f"{footage.width}x{footage.height}"
        )
        print(f"active spans: {[(s.start, s.end) for s in ingested.active_spans]}")
        print(f"proxy: {footage.proxy}")

        windows = tuple(
            SamplingWindow(span=span, fps=args.fps, concepts=tuple(args.concept))
            for span in ingested.active_spans
        )
        plan = SamplingPlan(footage_id=footage.id, windows=windows)
        for run in (1, 2):
            started = datetime.now(UTC)
            scope = BudgetScope(workspace_id="smoke", job_id=f"run-{run}-{started:%H%M%S}")
            result = await perception.runner.run(footage, plan, perception.video(scope))
            spent = await repository.spent_usd(started, "smoke")
            per_concept = {
                c: sum(o.concept == c for o in result.observations) for c in args.concept
            }
            print(
                f"run {run}: {result.paid_requests} paid, {result.cached_requests} cached, "
                f"observations {per_concept}, ledger ${spent}"
            )

        if args.box:
            t, x0, y0, x1, y1 = (float(v) for v in args.box.split(","))
            proxy = await perception.proxies.ensure(
                footage.content_hash, footage.original_key, ingested.active_spans, args.fps
            )
            request = SegmentRequest(
                footage_id=footage.id,
                media_key=proxy.proxy.blob_key,
                width=proxy.width,
                height=proxy.height,
                frame_times=proxy.frame_times,
                prompt=ObjectPrompts(
                    concept="tracked box",
                    objects=(
                        ObjectPrompt(t=t, object_id=1, target=Box(x0=x0, y0=y0, x1=x1, y1=y1)),
                    ),
                    direction="forward",
                ),
            )
            segmenter = perception.video(BudgetScope(workspace_id="smoke", job_id="box"))
            print(f"box prompt estimate: {segmenter.estimate(request)[0].quantity} frames")
            result = await segmenter.segment(request)
            scores = [round(o.score, 3) for o in result.observations]
            print(
                f"box prompt: {len(result.observations)} observations, "
                f"first scores {scores[:5]}, {result.charges}"
            )

    ledger_usd = await repository.spent_usd(EPOCH, "smoke")
    await repository.close()
    print(f"ledger (all smoke runs on this data root): ${ledger_usd}")
    print(f"provider-reported frames this process: {transport.reported_frames}")


if __name__ == "__main__":
    asyncio.run(main())
