# Rewind

A camera checker for production and packing lines that you talk to through chat. The user sends footage and writes rules in plain English. SAM 3.1 finds and tracks the items and their parts. Muse Spark turns each sentence into a strict rule. A deterministic engine reports every item that broke a rule, with a proof clip. Rules also run over footage already processed. Meta AI Developer Hackathon, Indie track; due Nov 4, 9:00 PM ET.

## Read first

- [docs/PLAN.md](docs/PLAN.md): what is done, what is next, in order. Work only on what it lists.
- [docs/RULES.md](docs/RULES.md): code layout and coding rules. Read before writing code.
- [docs/DECISIONS.md](docs/DECISIONS.md): why things are the way they are.
- [README.md](README.md): how to run it.

Keep these short and plain. No codes or ids for tasks; name things by what they are. If a line would not change what the next session does, delete it. Do not add more Markdown files.

## Product principles

- Every finding is provable: a violation always links to a clip, and an insight always links to the events behind it.
- Cost follows the rules, not the length of the footage. A paid call happens only when the plan asks for it, and never twice for the same input.
- AI translates and perceives; plain code decides.
- Every chat app (Telegram, WhatsApp, web, MCP) reaches the same agent and gets the same answers.
- Works for any line: a new line is a new sentence and new footage, never new code. Never tune logic or thresholds to one clip; check a fix on another kind of footage.
- Plain words first. Replies are read by a line manager, not an engineer: times instead of ids ("the box at 00:14"), one line per rule, boxes drawn on the proof, as few messages as possible.

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
