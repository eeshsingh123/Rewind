# Plan

The one place to see what is done, what is next, and in what order. Update it at the end of every work session. If a line would not change what the next session does, delete it.

Deadline: Nov 4, 9:00 PM ET (submit on the evening of Nov 4 IST).

## The demo story

A customer says "my box arrived without a label". The manager types that rule into Telegram. Rewind runs it over footage it already has and answers in plain words: when it started, how often, and a clip with the box drawn in red. Every answer has a clip behind it.

## Done

- Project setup: one check command (`uv run poe check`), CI on every PR, `main` protected.
- API checks: SAM 3.1 and Muse Spark called for real; prices and limits are in DECISIONS.md.
- Perception: a video is cut down to its busy parts and sent to SAM once; results are cached, so a rerun is free. Every paid call is logged and capped.
- Rules: a sentence becomes a strict rule; plain code judges every item. Works on busy lines (SAM's 16-object limit is handled).
- Chat (Telegram, WhatsApp built): send a video and a rule, get a live checklist, then the result with a proof clip. `/start` offers a sample clip. Rules are read back before they are used.
- Proof clips: boxes, short tags and a clock drawn on the video.
- Sim (Blender): built, then frozen. Not used for the demo.
- Checked live on the stock clip: "All 6 boxes that left the view had a label" (correct). First run 5 min, $0.08.

## Next, in order

1. **Demo footage with a real fault.** Pick 3–5 stock clips where an item visibly breaks a rule (missing label, open flap). The owner downloads them into `data/stock/` with their credits (candidates are in `data/stock/candidates/`).
   Done when: at least one clip shows a red box for a real fault in Telegram.
2. **Insights.** After a check, add at most a few lines like "first seen at 00:14, 5 times since" or "the line stopped for 40 s at 01:10", each with its clip. Show them only when they are notable; a normal run gets one quiet line.
   Done when: on a clip with a known fault, the reply gives the right first time and count, and each line links to a clip under 16 MB.
3. **Accuracy table.** Hand-label the demo clips (which item, what time, what fault) and score Rewind against them: precision, recall, cost, run time. One table for the README.
   Done when: the table is in the README.
4. **Submission.** README with a diagram, a 3-minute demo video, the written answers, credits. See the checklist below.

## Only if time is left

- A better cost estimate before a check (it said $0.01 for a run that cost $0.08).
- Crop to the work area, if a demo clip has more than 16 similar objects in a frame.
- A web page for upload and results.
- An MCP server with the same tools as the chat.
- Scene rules for shelves and docks ("this area must not be empty for more than 5 minutes").

If time runs short, cut from the bottom of that list up, then WhatsApp. Never cut: the proof clips, the accuracy table, one working chat channel.

## Submission checklist

- [ ] Public GitHub repo; check it opens logged out.
- [ ] 3-minute demo video on YouTube showing what was built; check it opens logged out.
- [ ] Everyone who worked on it listed on the submission page.
- [ ] Written answers: how we used Meta products, why Meta's models matter here, feedback on them. Feedback notes so far: SAM has no fps setting, a 50-frame minimum and one concept per request; usage is missing on background polls; Muse's prompt cache hit only with `prompt_cache_key`; SAM masks are slow to decode in Python.
- [ ] README credits: the Pexels stock clips, and the sim's assets (sim/CREDITS.md).
- [ ] Before judging: run the sample once so later taps are fast and free.

## Open questions for the owner

- Product name: keep "Rewind"?
- Judges try the live product. Is the Telegram bot plus a tunnel to the owner's machine enough, or do we need an always-on host?
- Rule edge cases, decided for now as written in DECISIONS.md: should "at the end of the footage" judge items that left earlier? Should a per-minute rate use fixed or sliding minutes?

## Owner setup still open

- [ ] WhatsApp keys (only if WhatsApp stays in the demo; the names are in `.env.example`).
- [ ] Postgres: not needed yet; SQLite works.
