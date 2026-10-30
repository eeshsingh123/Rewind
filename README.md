# Rewind

Ask your footage anything, from the web, Telegram or WhatsApp. Send a photo or a video and ask in plain words, by text or by voice note in any language, with no setup. Rewind shows each step as it works and answers with proof: a clip, or the photo with boxes drawn.

- **Muse Spark** looks at the footage, plans the steps and reasons.
- **SAM 3.1** finds and tracks the items.
- **Muse Voice** turns a voice note into the question.
- **Plain code** counts, times, draws the proof and checks every claim against what the tools found.

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

Start it with a public tunnel instead, for WhatsApp and the timeline links in chat replies (needs `cloudflared`):

```bash
uv run poe live
```

Then open http://127.0.0.1:8000 for the web page, or the bot in Telegram and send a photo or a video.

WhatsApp starts once its keys are set, with no code change: the token, phone number id, app id, app secret, verify token and business account id (names in `.env.example`). The token needs the `whatsapp_business_messaging` and `whatsapp_business_management` permissions. Start with `uv run poe live`; the app registers its webhook and subscribes the account itself.

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

Stock footage from Pexels (credits beside each clip in `data/stock/`). Sample shelf photo: ["Shelves with various milk boxes"](https://www.pexels.com/photo/shelves-with-various-milk-boxes-7451957/) by Nothing Ahead on Pexels. Sim assets are listed in [sim/CREDITS.md](sim/CREDITS.md).
