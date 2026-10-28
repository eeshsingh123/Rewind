# Memory

Last updated: 2026-10-28

## State

- Discussion done: product (D-1), architecture (D-2, D-3), stack (D-4), cost strategy (D-6), channels (D-8), sim bench (D-9), parallel tracks and session effort (D-11), local-first runtime without Modal (D-12).
- No code yet. Next: T0 Foundations and the SAM/Muse half of S0 (the key exists), in parallel. See MODULES.md for the order.
- Repo: github.com/eeshsingh123/Rewind, public, empty. T0 makes the first push, then protects `main` (GitHub cannot protect a branch that does not exist yet).

## Blockers the owner sets up

Needed for T0:
- [x] Empty GitHub repo `Rewind`. `gh` is signed in as eeshsingh123.
- [ ] `cloudflared` installed (`winget install Cloudflare.cloudflared`) for the tunnel check.

Needed for S0 (blocks T2, T3 compiler, T4):
- [x] Meta Model API key, in the user environment as `META_API_KEY`. Read it only through settings; never print it. Whether SAM 3.1 and Muse Spark are both enabled on it is checked in S0.
- [ ] The SAM 3.1 and Muse Spark API reference pages saved into `docs/vendor/` if dev.meta.ai needs a login to read them (endpoints, request/response shapes, pricing, limits).
- [ ] WhatsApp: Meta developer app with WhatsApp added, test number id, a **system user token** (the temporary token expires in 24 h), the owner's phone added as a test recipient.
- [ ] Telegram: bot token from @BotFather.
- [ ] Neon Postgres project and connection string.

Needed for T1:
- [ ] Blender 4.5 LTS installed (`winget install BlenderFoundation.Blender.LTS.4.5`, or from blender.org).
- [ ] Hugging Face token (SimReady download).

Needed later:
- [ ] Modal credit (optional, D-12): if it arrives, add the Modal adapters and entry point.
- [ ] Fireworks and/or Baseten credits requested (fallback language model).
- [ ] 2–3 free stock clips of real bottling or packaging lines picked (Pexels/Pixabay) for the sim-to-real check.
- [ ] The official submission rules: video length, whether the repo must be public, required Meta models.

## Machine

- Windows 11, GTX 1650 4 GB (driver 616.64), 8 GB RAM. Python 3.14 is global; the project pins 3.12 through uv (0.9.21). ffmpeg 8.0.1 runs from the Downloads folder (on PATH). Blender is not installed. Docker Desktop is installed.

## Open questions

- Product name (the repo is `Rewind`; the name can still change).
- Does the submission need a hosted URL, or are a demo video and the repo enough? Decides whether T8 needs a host (D-12).
- The demo's spoken language for any voice output (dropped from P0; only relevant if added back).
