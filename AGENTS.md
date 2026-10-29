# Rewind (working name)

A camera checker for production and packing lines that you talk to through chat. The user sends footage of a line and writes rules in plain English. SAM 3.1 finds and tracks the items and their parts. Muse Spark turns each sentence into a rule in a small, strict rule language. A deterministic rule engine reports every item that broke a rule, with a proof clip. Rules also apply retroactively to footage that was already processed ("Ctrl+F for your line"). Built for the Meta AI Developer Hackathon, Indie track. Submission is due Nov 4, 9:00 PM ET; submit on the evening of Nov 4 IST.

## Product principles

- Every finding is provable: a violation always links to a clip, and an insight always links to the events behind it.
- Cost follows the rules, not the length of the footage. A paid model call happens only when the sampling plan asks for it, and never twice for the same input.
- The rule engine is deterministic. AI translates and perceives; plain code decides.
- The chat app does not matter: WhatsApp, Telegram, web and MCP clients all reach the same agent and get the same answers.
- Works for any line: a new line is a new sentence and new footage, never new code. Never tune logic or thresholds to one clip; a fix found on one video is checked on another kind of footage.
- Plain words first: the user experience is the top priority. Replies and clips are read by a line manager, not an engineer: times instead of ids ("the box at 00:14"), one line per rule, boxes drawn on the proof, and as few messages as possible.

## Source of truth

- [docs/MODULES.md](docs/MODULES.md): the tracks and the scope of each. Do not build outside it.
- [docs/RULES.md](docs/RULES.md): design, code and git rules. Read before writing code.
- [docs/DECISIONS.md](docs/DECISIONS.md): architecture decisions. Only those that still guide upcoming work keep their why; settled ones shrink to one line under Done.
- [docs/MEMORY.md](docs/MEMORY.md): current state, active tracks, blockers, open questions. Update at the end of every work session.

Keep all four short. If a line would not change what the next session does, delete it.

## How we work

- The owner drives; the agent implements end to end and owns code quality.
- Explain the why behind every non-trivial choice. Say so when there is a better way than what was asked.
- Ask when a requirement is ambiguous. Do not guess on product behaviour.
- Never break existing behaviour. A change that touches many modules is a design smell: stop and raise it.
- Before reporting done: run the checks, and say what was and was not verified.

## Build protocol

A track earns its place only if it adds real value; cut anything that does not.

Each track goes through these stages, and each stage waits for the owner:
1. Discussion: understand the ask, look at the code as it is, present the approach, limits and tradeoffs. Push back where there is a better way.
2. Mockup (only for tracks the user sees: web UI and chat flows): an artifact showing how it looks, with the open questions on it.
3. Build: re-read the track's scope in MODULES.md and RULES.md, implement on a feature branch, verify, and open a PR. Ask when something comes up mid-build. Report what was added, why, and what is next.

## Story briefs for other sessions

Tracks run in parallel sessions. A brief must stand alone, without access to this project's conversations:
- context: what the product is and where this piece sits;
- exact scope and explicit out-of-scope;
- the packages to create or touch, and the contracts and ports to implement against;
- acceptance checks that can be run;
- a pointer to RULES.md.

A track must not change an existing contract. If it needs to, the contract changes first in its own PR.

## Session limits

- No sub-agents unless the owner asks.
- No code review and no review fixes unless the owner asks for one in that session.
- Read only the files the task needs. Run `uv run poe check` before every PR, and one live smoke run per track.
