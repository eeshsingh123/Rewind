# Rewind

A camera checker you talk to through chat. Send a video of a packing or production line and write a rule in plain English ("every box needs a label when it leaves the view"). Rewind finds every item that broke the rule and replies with a proof clip, the item boxed in red. Rules also run over footage it has already seen, like Ctrl+F for your line.

- **SAM 3.1** finds and tracks the items and their parts.
- **Muse Spark** turns each sentence into a strict rule and runs the chat.
- **Plain code** decides. The rule engine is deterministic, so every answer is repeatable and backed by a clip.

Built for the Meta AI Developer Hackathon, Indie track.

## Run it

Needs [uv](https://docs.astral.sh/uv/) and ffmpeg on the PATH.

```bash
uv sync
```

Copy `.env.example` to `.env` and fill in at least `META_API_KEY` and the Telegram settings (`REWIND_TELEGRAM_BOT_TOKEN`, `REWIND_TELEGRAM_ALLOWED_USER_IDS`). A channel whose token is unset is not started.

Start the app (Telegram works without a public URL):

```bash
uv run poe serve
```

Start it with a public tunnel instead, for WhatsApp and upload links (needs `cloudflared`):

```bash
uv run poe live
```

Then open http://127.0.0.1:8000 for the web page, or the bot in Telegram and send `/start`.

## Develop

Lint, format, types, import rules and tests in one go:

```bash
uv run poe check
```

Live smoke runs against the real APIs (they cost money and never run in CI) are in `scripts/`.

## Docs

- [docs/PLAN.md](docs/PLAN.md): what is done, what is next, in order.
- [docs/DECISIONS.md](docs/DECISIONS.md): why things are the way they are.
- [docs/RULES.md](docs/RULES.md): code layout and coding rules.

## Credits

Stock footage from Pexels (credits beside each clip in `data/stock/`). Sim assets are listed in [sim/CREDITS.md](sim/CREDITS.md).
