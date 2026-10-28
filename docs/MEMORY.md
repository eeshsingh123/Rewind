# Memory

Last updated: 2026-10-28

## State

- Discussion done: product (D-1), architecture (D-2, D-3), stack (D-4), cost strategy (D-6), channels (D-8), sim bench (D-9), parallel tracks and session effort (D-11), local-first runtime without Modal (D-12), Muse model (D-14).
- S0 is done for the Meta models: SAM 3.1 and Muse Spark are both enabled, and D-6 now has the confirmed figures. Findings and recordings are in the uncommitted `spikes/s0/`; T2 copies the recordings it needs into `tests/recordings/`. The WhatsApp and Telegram checks are still open.
- T0 Foundations is built on `t0-foundations`, with its PR open and waiting for the owner to merge. It has Contracts v1, the seven core ports (`Segmenter` and `LanguageModel` already shaped by S0: object prompts, the 50-frame floor, tools, images and effort), settings, `/health`, and `uv run poe check` (ruff, pyright, import-linter, pytest) running locally and in CI. Contract conventions are in D-13 (accepted).
- `main` is protected: a PR is required (admins too) and force-push is blocked. The `check` status is required.
- Contracts v1 also has the `Line`, `Footage` and `Rule` records and their `Repository` methods, so T2 (footage), T3 (rules) and T4 (lines) need no contract PR.
- Next: after T0 merges, start T3 Rules (high) and T2 Perception (medium) first, at most two sessions at once (D-11), each with a standalone brief (AGENTS.md). Then T1 Sim (needs the Hugging Face token), then T4 Chat (mockup first; needs the Telegram, WhatsApp and Neon tokens). T1 can keep pydantic out of Blender: `sim/render.py` runs under the project's Python and drives Blender as a subprocess; only the builders run inside Blender, and the ground truth is validated against the contracts on the project side.

## Blockers the owner sets up

Needed for T0:
- [x] Empty GitHub repo `Rewind`. `gh` is signed in as eeshsingh123.
- [x] `cloudflared` 2026.10.5 installed at `C:\Program Files (x86)\cloudflared\cloudflared.exe` (open a new shell for PATH). `/health` returned 200 through a quick tunnel: `uv run poe serve`, then `cloudflared tunnel --url http://localhost:8000`.

Needed for S0 (the Meta half is done; the channel half blocks T4):
- [x] Meta Model API key, in the user environment as `META_API_KEY`. Read it only through settings; never print it. SAM 3.1 and Muse Spark are both enabled on it.
- [x] API reference: public, no login, at `https://dev.meta.ai/docs/*.md` (index `https://dev.meta.ai/llms.txt`).
- [ ] WhatsApp: Meta developer app with WhatsApp added, test number id, a **system user token** (the temporary token expires in 24 h), the owner's phone added as a test recipient.
- [ ] Telegram: bot token from @BotFather.
- [ ] Neon Postgres project and connection string.

Needed for T1:
- [x] Blender 4.5 LTS installed with winget (see Machine for the path).
- [ ] Hugging Face token (SimReady download).

Needed later:
- [ ] Modal credit (optional, D-12): if it arrives, add the Modal adapters and entry point.
- [ ] Fireworks and/or Baseten credits requested (fallback language model).
- [ ] 2–3 free stock clips of real bottling or packaging lines picked (Pexels/Pixabay) for the sim-to-real check.
- [ ] The official submission rules: video length, whether the repo must be public, required Meta models.

## Machine

- Windows 11, GTX 1650 4 GB (driver 616.64), 8 GB RAM. Python 3.14 is global; the project pins 3.12 through uv (0.9.21). ffmpeg 8.0.1 runs from the Downloads folder (on PATH). Blender 4.5 LTS: `C:Program FilesBlender FoundationBlender 4.5blender.exe` (4.5.10, not on PATH; T1 reads it from a setting). Docker Desktop is installed.

## Open questions

- Product name (the repo is `Rewind`; the name can still change).
- Does the submission need a hosted URL, or are a demo video and the repo enough? Decides whether T8 needs a host (D-12).
- The demo's spoken language for any voice output (dropped from P0; only relevant if added back).
