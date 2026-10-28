# Tracks

The roadmap. Each track is one branch and one PR, with one commit series per sub-module. Tracks run in parallel only when they touch different packages (D-11). A track handed to another session gets its own self-contained brief (AGENTS.md), written when it is assigned.

## Package layout

```
src/rewind/
  contracts/   frozen Pydantic models shared by everything; depends on nothing but pydantic
  core/        domain logic and ports (Protocols); depends only on contracts
    perception/  sampling planner, strategies registry, idle-skip and tripwire logic over frames
    rules/       rule language, predicate registry, engine, zones, compiler (via the LanguageModel port)
    evidence/    clip specs, order linking, insights stats
    conversation/ agent loop, tools, canonical messages
    budget/      ledger, estimator, kill switch
  adapters/    one module per external system, each implementing a core port
    segmenters/  meta_video, meta_image
    models/      muse, fireworks
    storage/     local_disk, postgres (modal_volume later, D-12)
    video/       ffmpeg, opencv
    jobs/        in_process (modal later, D-12)
    channels/    telegram, whatsapp, web, mcp
  app/         composition root: settings, wiring, FastAPI routes, worker start-up
sim/           Blender scenario renderer and builders; depends on contracts only
eval/          scoring and sweeps; depends on contracts and core
tests/         focused tests and recordings/ of real API responses
```

Dependencies point inward: `app -> adapters -> core -> contracts`. `sim` and `eval` never import `adapters` or `app`.

## Vocabulary

- Line: one camera's view of a production or packing line, in a workspace.
- Footage: one uploaded or streamed video of a line. Kept as the original plus a proxy.
- Concept: a short noun phrase SAM looks for ("cardboard box", "shipping label").
- Observation: one detection of a concept in one frame (box, optional mask, score, track id).
- Entity: the thing a rule is about (a carton, a jar, a pallet). Parts are what it must have.
- Zone: an area or tripwire in the image: drawn, found by text, or the implicit "leaves the view".
- Rule: a sentence compiled to the rule language (`RuleSpec`).
- Violation: one entity breaking one rule at one time, with evidence.
- Sampling plan: which concepts, regions, fps and windows perception pays for (D-6).
- Ledger: the record of every paid call (D-7).

## Contracts v1 (built in T0)

- `ScenarioConfig`, `GroundTruth` (sim): items with `id, order_id, t_spawn, t_cross{zone}, props{...}, expected_violations[]`, events (jams), camera, projected zones.
- `Observation`, `Track`, `Zone`, `SamplingPlan`.
- `RuleSpec`: `entity{key, phrase}`, `parts{key: phrase}`, `checkpoint` (`enters(zone)` | `exits_view` | `end_of_footage`), `requirements[]` (predicate `has | in_zone | count_at_least`, with timing `at_checkpoint | ever_before`), `order[]` (`before(a, b)`), `dwell{zone, max_s}`, `rate{min_per_minute}`, `unsupported[]`. Predicates are a discriminated union; adding one is additive.
- `Event`, `Violation`, `Insight`.
- `CostEntry`, `CostEstimate`.
- `CanonicalMessage` (in), `Reply` (out), `ChannelCapabilities`.

## Tracks

Sizes are focused hours.

### S0 API checks (spike, no merge) · 3–5h · needs keys
Call SAM 3.1 (video and image) and Muse Spark for real. Record the raw responses into `tests/recordings/`. Answer:
- SAM billing: per frame per prompt, or per frame for all prompts? Frames uploaded, or frames processed? Is there an fps/stride setting?
- SAM limits: video length and size, sync or async, rate limits, output shape (boxes, RLE masks, track ids, scores).
- Muse Spark: base URL and model names, JSON or structured output, image input, tool calling, prompt caching, price per token.
- WhatsApp: send and receive a video on the test number. Telegram: the same through a bot.

Output: the answers in DECISIONS.md (D-6 figures confirmed or revised), and the recordings.

### T0 Foundations · 5–7h · first, blocks everything else
- uv project (Python 3.12), ruff, pyright, pytest, import-linter, a poe `check` task, and GitHub Actions running `check`.
- Contracts v1 (above), and the core ports: `Segmenter`, `LanguageModel`, `BlobStore`, `Repository`, `VideoTools`, `JobRunner`, `Channel`.
- Settings from the environment (`.env` locally, never committed); a composition root with a FastAPI app under uvicorn answering `/health`; a `poe serve` task.
- Acceptance: `uv run poe check` is green in CI, and `/health` returns 200 both on localhost and through a `cloudflared` quick tunnel.

### T1 Sim bench · 20–28h · depends on T0 contracts · runs locally in Blender
- `sim/render.py` turns a scenario YAML + seed into `video.mp4`, `ground_truth.json` and `zones.json`.
- Builders: conveyor (scrolling belt), jar + cap + label, carton with 4 hinged flaps + tape + QR label per order, stations with fault rates, jam events.
- Selective SimReady fetch, CC0 materials and HDRIs, randomization (lights, camera, speed, faults, colours), CCTV degradation.
- Realistic textures and props, not flat shapes: in S0, SAM concept prompts were weak on flat synthetic shapes, so a plain sim would measure the sim instead of the pipeline.
- Scenarios: S1 jar line and S2 carton packing (P0); S3 outbound dock (P2).
- Out: physics, humans, Isaac Sim.
- Acceptance: two seeds of S1 and S2 render at 720p, and their ground truth matches a manual check of 10 items.

### T2 Perception · 14–20h · depends on T0, S0
- Ingest: proxy transcode, time map, content hash. Idle skip. `Segmenter` adapters for Meta video and image. Observation cache in Parquet.
- Planner with the `dense_track` strategy. The ledger and budget guard are wired into every paid call.
- P1: the `checkpoint` strategy and coarse-to-fine sampling. P2: crop-and-pack, self-hosted SAM.
- Acceptance: an S1 render produces observations through the Meta video adapter; a rerun costs $0 (cache hit); the ledger matches the provider's usage.

### T3 Rules · 10–14h · depends on T0 (engine), S0 (compiler)
- Predicate registry and evaluators, an engine over observations (`at_checkpoint`, `ever_before`, order, dwell, rate, count), zones (implicit exit, found by text through the `Segmenter`, drawn).
- Compiler: Muse Spark → `RuleSpec`, validated with retries, with `unsupported` filled in. Compiled rules are cached by normalized text.
- Acceptance: engine tests on fixture observations cover each requirement kind; the example sentences from S1 and S2 compile to the expected specs.

### T4 Chat gateway · 12–16h · depends on T0, S0 · mockup first
- Canonical messages; Telegram and WhatsApp adapters with dedupe; the agent and its tools (`create_line`, `upload_link`, `add_rule`, `list_rules`, `analyze` with an estimate and confirmation, `violations`, `get_clip`, `insights`); signed upload links; async progress and result replies; `/link` codes to join channels.
- P1: an MCP server exposing the same tools.
- Acceptance: on both Telegram and WhatsApp, sending a clip and a rule returns a violation reply with a clip. Sending the same webhook twice processes it once.

### T5 Evidence and insights · 8–12h · depends on T2, T3
- Clips with masks or boxes drawn on, cut from the proxy; QR decode → order link; cycle-time normal ranges, bottleneck and jam stats in DuckDB; a Muse Spark narrative that cites clips (P1).
- Acceptance: each S2 violation has a playable clip under 16 MB, and "order 1043" returns that order's clip.

### T6 Web · 8–12h · depends on the core API · mockup first
- Upload page (also the target of the signed upload links), results timeline, clip player, rule entry. P1: zone drawing, eval dashboard, cost meter.

### T7 Evaluation · 6–8h · depends on T1, T2, T3
- Score against ground truth (D-10). P1: the sweep and the cost-vs-accuracy chart. Hand-labelled stock clips scored separately.

### T8 Submission · 6–8h · last
- Demo video, README with diagram, Meta models used, results table, limits, and credits and licenses (SimReady CC-BY-4.0, CC0 assets, stock clips).

## Order

1. Now: **T0**, and **S0** as soon as the keys exist.
2. After T0 is merged: **T1, T2, T3, T4** in parallel (T4 starts with its mockup).
3. Then T5, T6, T7. Last, T8.

If time runs short, cut in this order: S3 → MCP → crop-and-pack and self-hosted SAM → narrative insights → zone drawing → eval dashboard. Never cut the eval table, the proof clips, or one working chat channel.
