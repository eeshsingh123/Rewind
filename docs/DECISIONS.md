# Decisions

Only decisions that still guide upcoming work keep their why. Everything settled and built is one line under Done. Numbers are never reused.

## Guiding upcoming work

### D-1 Product: retroactive rules, with packing as the lead story — accepted (2026-10-28)

Rules in plain English run over footage, including footage processed before the rule existed. The lead demo is an e-commerce carton line answering "the box arrived empty". A jar line and a warehouse dock prove it works on any line.
Why: a generic "SAM plus an LLM watches the camera" build will be common at this hackathon. Retroactive rules with proof clips stand out, are easy to demo, and come straight from the architecture in D-2.

### D-2 Perception is separate from rules; the engine is deterministic — accepted (2026-10-28)

Perception (SAM 3.1) turns footage into observations once per input and concept, and stores them. Rules are compiled into the rule language and evaluated by plain Python over the stored observations, any number of times.
Why: this makes retroactive rules cheap, keeps decisions testable and repeatable, and avoids a vision model guessing over long videos. Muse Spark is used only to translate rules, to give a yes/no on borderline crops (D-6), and to write narratives over computed stats.

### D-3 Ports and adapters with one composition root — accepted (2026-10-28)

Package layout: `contracts` (frozen Pydantic models) ← `core` (domain logic and ports as Protocols) ← `adapters` (Meta SAM, Muse, Fireworks, local disk, Postgres, ffmpeg, in-process jobs, WhatsApp, Telegram, MCP) ← `app` (FastAPI routes, entry points, wiring). `sim` and `eval` depend only on `contracts` and `core`.
Why: the owner's open/closed rule. A new channel, model or storage is a new adapter. import-linter enforces the direction.

### D-4 Stack — accepted (2026-10-28)

Python 3.12 + uv, FastAPI under uvicorn on the owner's machine (D-12), Neon Postgres via SQLAlchemy 2.0 (Core) with Alembic, Parquet + DuckDB for observations, ffmpeg and OpenCV, HTMX + plain JS for the web, the official `mcp` SDK, Blender 4.5 LTS for the sim.
Why: the ML and video tooling is Python-first; Postgres is safe with parallel workers and reachable from any later host.

### D-5 Storage: local disk first, behind BlobStore — revised (2026-10-28)

Blobs (footage, proxies, keyframes, clips, Parquet) live on local disk behind the `BlobStore` port, under one data root from settings. Media is served by our own signed, expiring URLs. A Modal Volume or Cloudflare R2 is a later adapter.
Why: no extra account or payment card before Nov 4, and swapping storage changes one adapter.

### D-6 Cost strategy — accepted, figures confirmed by S0 (2026-10-28)

Prices (S0): SAM 3.1 video $0.20 per 1k frames, image $2.50 per 1k images. SAM takes one concept per request and bills every frame of the file sent (or of the tracked range for box prompts), with a 50-frame minimum per request, found or not. It has no fps setting: the proxy's fps is the cost knob. The ledger reads `usage.video_frames_processed` / `images_processed`. Muse Spark 1.3 costs $1.25 in, $0.15 cached and $4.25 out per 1M tokens; Contributor costs $0.10, $0.002 and $0.20 (D-14). Reasoning tokens bill as output and dominate small calls.
Use video wherever possible. Each step is kept only if the eval sweep (D-10) shows it doesn't cost accuracy. In build order:
1. Re-encode at upload to a 720p proxy at the planned fps, keeping a map back to the original time. Mandatory: the file's frame count is the bill.
2. Skip idle footage on the CPU (MOG2 inside the zones); upload only the active spans, joined together.
3. Rule-aware sampling planner: the compiled rules decide the concepts, regions, fps and windows.
4. Concept scoping: track only the entity continuously; look for parts only near checkpoints.
5. `checkpoint` strategy for flow lines: tripwire crossings found on the CPU, item identity from our own association (SAM object ids are reused across items), falling back to `dense_track` for a window when the order is ambiguous. Checkpoint windows are joined into one clip per concept, or box-prompted up to 16 objects at a time and tracked `forward`; never one request per item (the 50-frame floor makes that about 10× dearer).
6. Coarse-to-fine: sample more densely only around borderline decisions.
7. Muse Spark yes/no on a crop only when a decision is borderline, at `minimal` effort, within a budget per job.
8. Cache by `(content_hash, frame_idx, concept, model_version)`. SAM results vary slightly between runs, so the cache is the source of truth for replays. A retroactive rule with a new concept runs on stored keyframes only.
9. Crop-and-pack and self-hosted SAM: experiments only, after the hackathon unless the sweep makes a strong case.

Estimate for one camera, an 8-hour shift, about 4,000 items and 3 concepts: about $1,080 a day with naive image calls, about $86 a day with naive video calls at 5 fps, and about $5–12 a day with every step above (entity at 2 fps on active spans about $6, parts in joined checkpoint windows about $6). Sending one request per item instead would cost about $80 a day. Muse yes/no checks on Contributor cost about $0.07 for 500.

### D-7 Budget guardrails — accepted (2026-10-28)

Every paid call writes a ledger entry. A cost estimate is shown before every analysis; chat asks for confirmation above a threshold. Each workspace has a daily cap, and a global kill switch stops paid calls at $120. Webhooks are deduplicated by provider message id and acknowledged with 200 at once, with the work done in a job. The agent has caps on tool calls and tokens per turn; expensive tools are idempotent.
Why: the owner wants no runaway spend. Webhook retries and agent loops are the usual ways costs blow up.

### D-8 One agent behind every channel — accepted (2026-10-28)

Channel adapters turn provider webhooks into a `CanonicalMessage` and render replies within each channel's capabilities (media size limits, buttons). One agent (Muse Spark with tool calling, with a Fireworks fallback) owns the conversation. Long footage arrives through a signed upload link. The same tools are exposed over MCP. Channels for the hackathon: Telegram, WhatsApp, web, MCP. No Muse-specific work (owner: keep it centralized).

### D-9 Simulated test bench in Blender — accepted (2026-10-28)

Blender 4.5 LTS run headless with Eevee renders scenarios from YAML into a video, `ground_truth.json` and `zones.json`. All motion is keyframed, with no physics and no humans; the lines are automated (capper, labeller, taper). SimReady Warehouse-01 props (CC-BY-4.0) are downloaded selectively, and their materials are replaced with CC0 ambientCG textures and Poly Haven HDRIs. The conveyor, cartons, flaps, tape and QR labels are built from simple shapes. Renders are made to look like CCTV: blur, noise, shake, H.264 compression.
Why: no factory access; an exact answer key; Isaac Sim needs an RTX card and is too slow to learn in time.
Owner hardware: GTX 1650 4 GB and 8 GB RAM. This is enough for Eevee at 960×540–1280×720 with textures at 2K or smaller, rendering one job at a time. Too slow → lower the resolution or the sample count. Neither SAM nor Isaac Sim runs locally.

### D-10 Evaluation is the proof — accepted (2026-10-28)

Score per-item violation precision, recall and F1 (a match is the same item within ±2 s), dwell/jam timing, QR order-link accuracy, cost per 1,000 items, and latency. Sweep strategy × fps × gating to produce a cost-vs-accuracy curve. Hand-labelled stock clips are reported separately as the sim-to-real check. Never cut.

### D-11 Hackathon adaptations to the build protocol — accepted (2026-10-28)

Tracks run in parallel sessions, one branch and one PR each, as long as they touch different packages. Contracts change only in their own PR. The mockup stage applies only to tracks the user sees (web UI, chat flows).
Why: about 100 focused hours remain before Nov 4, and building one track at a time would not finish.
Sessions run on Opus 5.5 medium, except T0 and the T3 engine, which run on high because every other track builds on their contracts and design. At most two sessions run at once, to stay inside the owner's $20 plan limits.

### D-12 Local-first runtime; Modal is a later deploy target — accepted (2026-10-28)

Everything runs on the owner's machine: the FastAPI app under uvicorn, blobs on local disk (D-5), and background work through a `JobRunner` port with an in-process worker. Telegram uses long polling, so it needs no public URL. WhatsApp webhooks, signed upload links, and any media URL the Meta API must fetch go through a Cloudflare quick tunnel (`cloudflared`, free, no account) to localhost. Modal (Volume, jobs, web endpoint) comes back later as adapters plus one entry point in `app`, with no change to `core`.
Why: the Modal credit has not arrived and waiting costs build days. SAM 3.1 and Muse Spark are hosted APIs, so no GPU is needed; ffmpeg, OpenCV and Blender already run locally.
Limits: the machine must stay on during live demos; the quick-tunnel URL changes on each restart, so the WhatsApp webhook is re-registered at start-up; 8 GB RAM means one heavy job (render or analysis) at a time. If the submission needs a hosted URL, T8 picks Modal (if credit arrives) or a Docker host.

### D-13 Contract conventions (T0) — accepted (2026-10-28)

- Boxes, points and zones are normalized to the frame (0..1), so they survive a proxy at another resolution. A mask is an RLE of the crop under its box, which is how SAM 3.1 returns it.
- `Observation.object_id` is the segmenter's own id, stable only while the object stays in view. Item identity is our `Track`, built by association (S0: SAM reuses ids for different cartons).
- `Segmenter` takes one prompt (one concept, or up to 16 marked objects) and one media blob plus its time map (the footage time of each frame). It bills every frame it is sent, so the caller sends a proxy at the planned fps. Adapters return `Charge`s read from the provider's reported usage, and estimate from price, so the ledger matches the bill (D-7).
- Contracts forbid unknown fields, so language model JSON with an invented key fails validation and is retried. Collections are tuples.
Why: these choices are shared by T1–T5; changing one later means a contract PR that blocks those tracks.

### D-14 Muse model: `muse-spark-1.3-contributor` by default — accepted (2026-10-28)

The language model id is the setting `REWIND_MUSE_MODEL`, which defaults to `muse-spark-1.3-contributor`. `muse-spark-1.3` is the drop-in alternative for footage that must not be used for training; switching is a change to the setting only.
Why: it has every feature we need (S0) and costs about 12–20× less. Limits: Meta may train on everything sent to it (fine for sim and stock footage), and it allows 150 requests per minute per team, so bursts of yes/no checks queue and back off on 429.

## Done

(nothing yet)
