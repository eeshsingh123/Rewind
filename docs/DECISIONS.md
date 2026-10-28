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

### D-6 Cost strategy — accepted, figures pending the S0 spike (2026-10-28)

Prices: SAM 3.1 video $0.20 per 1k frames, image $2.50 per 1k images. Use video wherever possible. Each step is kept only if the eval sweep (D-10) shows it doesn't cost accuracy. In build order:
1. Re-encode at upload to a 720p proxy at the planned fps, keeping a map back to the original time.
2. Skip idle footage on the CPU (MOG2 inside the zones); upload only the active spans, joined together.
3. Rule-aware sampling planner: the compiled rules decide the concepts, regions, fps and windows.
4. Concept scoping: track only the entity continuously; look for parts only near checkpoints.
5. `checkpoint` strategy for flow lines: tripwire crossings found on the CPU, item identity from conveyor order plus timing, falling back to `dense_track` for a window when the order is ambiguous.
6. Coarse-to-fine: sample more densely only around borderline decisions.
7. Muse Spark yes/no on a crop only when a decision is borderline, within a budget per job.
8. Cache by `(content_hash, frame_idx, concept, model_version)`. A retroactive rule with a new concept runs on stored keyframes only.
9. Crop-and-pack and self-hosted SAM: experiments only, after the hackathon unless the sweep makes a strong case.

Rough estimate for one camera, an 8-hour shift, about 4,000 items and 3 concepts: about $1,080 a day with naive image calls, about $86 a day with naive video calls at 5 fps, and about $2–10 a day with every step above. This depends on S0's billing answers: is billing per frame per prompt, and on frames uploaded or frames processed?

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

## Done

(nothing yet)
