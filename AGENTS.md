# Rewind

Ask your footage anything, from the web, Telegram or WhatsApp. The user uploads a photo or a video and asks in plain words, with no setup. Muse Spark looks at it, plans the steps and calls tools: SAM 3.1 to track things, crops to look closer, and plain code to count, time and draw the proof. Each step shows live, and every claim comes with proof. Meta AI Developer Hackathon, Indie track; due Nov 4, 9:00 PM ET.

## Read first

- [docs/PLAN.md](docs/PLAN.md): what is done, what is next, in order. Work only on what it lists.
- [docs/RULES.md](docs/RULES.md): code layout and coding rules. Read before writing code.
- [docs/DECISIONS.md](docs/DECISIONS.md): why things are the way they are.
- [README.md](README.md): how to run it.

Keep these short and plain. No codes or ids for tasks; name things by what they are. If a line would not change what the next session does, delete it. Do not add more Markdown files.

## Product principles

- Every finding is provable: each claim links to the evidence behind it, with a clip or a photo with boxes drawn where it has one.
- No setup: upload and ask. No lines or rules to create first.
- Use Muse fully: it watches, plans and reasons. SAM and code make per-item counts and times exact, and code checks every claim against tool results.
- A paid call never happens twice for the same input. SAM only when a question needs it. Always `muse-spark-1.3-contributor`.
- Every channel (web, Telegram, WhatsApp) reaches the same agent, shows the same live steps and gets the same answers.
- Works for any footage: never new code for a new scene. Never tune logic or thresholds to one clip; check a fix on another kind of footage.
- UI is clean and minimal, in warm, calm neutral colours.
- Plain words first. Replies are read by a line manager, not an engineer: times instead of ids ("the box at 00:14"), one line per finding, boxes drawn on the proof, as few messages as possible.

## How we work

- The owner drives; the agent implements end to end and owns code quality.
- Explain the why behind non-trivial choices. Say so when there is a better way than what was asked.
- Ask when a requirement is ambiguous. Do not guess on product behaviour.
- Never break existing behaviour. A change that touches many modules is a design smell: stop and raise it.
- For each piece of work: agree the approach with the owner, show a mockup first if the user will see it (chat flows, web pages), then build on a branch, verify, and open a PR.
- Before reporting done: run the checks, and say what was and was not verified.
- At the end of a session, update docs/PLAN.md.

## Session limits

- No sub-agents unless the owner asks.
- No code review and no review fixes unless the owner asks for one in that session.
- Read only the files the task needs. Run `uv run poe check` before every PR, and one live smoke run per piece of work.
- A brief for another session must stand alone: what the product is, the exact scope and what is out of scope, the files to touch, checks that can be run, and a pointer to RULES.md.
