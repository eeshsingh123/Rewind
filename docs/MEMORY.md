# Memory

Last updated: 2026-10-29

## State

- Discussion done: product (D-1), architecture (D-2, D-3), stack (D-4), cost strategy (D-6), channels (D-8), sim bench (D-9), parallel tracks and session effort (D-11), local-first runtime without Modal (D-12), Muse model (D-14).
- S0 is done for the Meta models. Findings and the raw recordings are in the git-ignored `spikes/s0/` (only in the main checkout). The WhatsApp and Telegram checks happen in T4's live smoke.
- `main` is protected: a PR is required (admins too), force-push is blocked, and the `check` status is required.
- On `main`:
  - T0 Foundations (PR #1): Contracts v1, the seven core ports, settings, `/health`, `uv run poe check` locally and in CI. Conventions in D-13.
  - T2 Perception P0 (PR #2, D-15): ingest with proxy and time map, idle skip, SAM 3.1 video and image adapters, Parquet observation cache, `dense_track` planner, ledger and budget decorators, `LocalDiskBlobStore`. Wiring: `rewind.app.perception.build_perception(settings, httpx_client)`, then `perception.video(BudgetScope(workspace_id, job_id))`. Smoke: `uv run python scripts/smoke_perception.py CLIP --concept X`. Mask decoding costs about 0.1 s per mask of CPU.
  - T3 Rules (PR #3, D-16): the deterministic engine with a predicate registry, zones (`view_exit`, drawn, `find_zone()` by text), `RuleCompiler` (cached, with `zone_phrases`) and `MuseLanguageModel`. Smoke: `uv run python scripts/smoke_rules.py`; re-record `tests/recordings/muse/compile_examples.json` whenever the compile prompt or schema changes. Its 4 open questions are under Open questions.
- Open PRs, in merge order: #7 rule exit semantics (D-16 revised), #8 SAM object cap (D-17), then #5 T4 Chat. #7 and #8 are independent, and #5 already contains both through a merge; after they land, merge `main` into #5 so its diff shows only T4. #6 T1 Sim is independent.
- T4 (#5) after the first live run's fixes: parts needed only at a checkpoint are looked for on one still per item (`core/analysis/parts.py`); proof clips draw the flagged item in red with a short tag, other judged items in red or green, the parts found, and a clock (`adapters/video/proof_clip.py`, a slice of T5 brought forward); one plain line per rule plus a "Show clips" button; a clip, its caption and its button go as one message; jobs start after the reply that announces them; a rerun replaces that rule's findings. Mockup: https://claude.ai/artifact/6TTGFqUvVdzidBVGUs66YE.
- Checked by eye on 2026-10-29: the stock clip reports "All 6 boxes that left the view passed", which is right (every parcel there has a label). A run takes about 3–4 minutes, mostly SAM requests and mask decoding.
- Neither sim render can show a true violation yet. In S2 the tape is a faint strip SAM cannot see, so every carton fails the tape rule, and 16 carton-like objects per frame (the shelf boxes count) cap tracking. In S1 SAM finds no jars at all (too small at 720p). The red-box demo moment needs T1 realism fixes, a real clip with a visible fault, or both.
- The first, unfinished T1 and T4 sessions and the done T2 and T3 sessions can be archived once their work is merged. The old local branches `claude/t1-sim-646423` and `feat/t4-chat` are superseded by the new sessions' branches.
- Track sessions do not edit MEMORY.md or DECISIONS.md. They put state and proposed decisions in the PR description, and the coordinator merges them in.
- A real stock clip for smoke runs and demos: `data/stock/pexels_5903898_packages_conveyor_720p.mp4` in the main checkout (credit in `CREDITS.txt` beside it; credit it in the README).

## Blockers the owner sets up

- [x] GitHub repo `Rewind`; `gh` signed in as eeshsingh123.
- [x] `cloudflared` at `C:\Program Files (x86)\cloudflared\cloudflared.exe`. Quick tunnel: `uv run poe serve`, then `cloudflared tunnel --url http://localhost:8000`.
- [x] Meta Model API key as the user variable `META_API_KEY`; `HF_TOKEN` likewise. Both are user-level, so a session started before they were set may not see them: in PowerShell, `$env:NAME = [Environment]::GetEnvironmentVariable('NAME','User')`. Never print them.
- [x] Blender 4.5 LTS (see Machine).
- [x] Telegram, needed for T4's live smoke: `REWIND_TELEGRAM_BOT_TOKEN` and `REWIND_TELEGRAM_ALLOWED_USER_IDS`, as user variables.
- [ ] WhatsApp, second channel: `REWIND_WHATSAPP_TOKEN` (system user, never expires), `REWIND_WHATSAPP_PHONE_NUMBER_ID`, `REWIND_WHATSAPP_APP_ID`, `REWIND_WHATSAPP_APP_SECRET`, `REWIND_WHATSAPP_VERIFY_TOKEN` (any string), `REWIND_WHATSAPP_ALLOWED_NUMBERS`. The owner's phone is a test recipient and has messaged the test number once.
- [ ] Neon Postgres: not needed until after T4 (SQLite for now; switching is a URL change plus Alembic).
- [ ] Optional: Modal credit (D-12), and Fireworks or Baseten credits for a fallback model.

## Submission (T8)

Judges use three things: the demo video, the code, and the live product.
- Public GitHub repository (check it opens logged out).
- A 3-minute demo video on YouTube (`https://youtube.com/watch?v=...`), showing the features, code and functionality built during the hackathon. Check the link opens logged out.
- Everyone who worked on the project listed on the submission page.
- Written answers: how we used Meta products; why Meta's models matter to the solution; feedback on using Meta products. Feedback material from S0 and the tracks: SAM has no fps setting, a 50-frame floor and one concept per request; usage is missing on background polls; the compile prompt hit the cache only with `prompt_cache_key`; SAM masks are slow to decode in Python.

## Machine

- Windows 11, GTX 1650 4 GB (driver 616.64), 8 GB RAM. Python 3.14 is global; the project pins 3.12 through uv (0.9.21). ffmpeg 8.0.1 runs from the Downloads folder (on PATH). Blender 4.5 LTS: `C:\Program Files\Blender Foundation\Blender 4.5\blender.exe` (4.5.10, not on PATH; `REWIND_BLENDER_PATH`). Docker Desktop is installed.

## Next

1. Merge #7 and #8, then #5 once `main` is merged into it, and this docs PR. Review #6 (T1).
2. Footage with a true violation for the demo: T1 realism (visible tape, larger jars, fewer box-like props behind the belt; re-run its SAM realism probe), and a real stock clip with a visibly unlabelled or untaped parcel (the owner downloads it).
3. T2 P1 region cropping, so cluttered scenes stay under SAM's 16 objects per frame (D-17).
4. Then T5 Evidence (masks on clips, QR order links, insights) and T7 Evaluation, two at a time. T6 Web after that, mockup first.
5. T8 Submission last: README, demo video and the written answers.

## Open questions

- Product name (the repo is `Rewind`; the name can still change).
- Judges look at the "live product". Is a working Telegram bot plus a tunnel to the owner's machine during judging enough, or does T8 need an always-on host (D-12)?
- T3's open questions (D-16 uses the current answers until the owner decides): `end_of_footage` for items that left earlier, the "4 flaps before tape" approximation, fixed or sliding rate windows. (Whether `exits_view` needs a frame edge: yes, D-16 revised.) T1 raised the first one too.
