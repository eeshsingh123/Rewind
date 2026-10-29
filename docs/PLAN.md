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
- Sim (Blender): built, then frozen. Two rendered clips with answer keys are now the demo footage (see below); nothing new is built in `sim/`.
- Checked live on the stock clip: "All 6 boxes that left the view had a label" (correct). First run 5 min, $0.08.

## Next, in order

1. **Demo footage with faults.** Two sim clips, 60 s each, with answer keys:
   - `data/sim/s1_jar_line-seed2`: 27 jars; 4 missing a label (leaving the view at about 00:08, 00:20, 00:43, 00:48), 1 missing a cap, 1 held too long at the capper; the belt stops 00:20.8–00:33.4.
   - `data/sim/s2_carton_packing-seed2`: 25 cartons; 3 without tape, 3 without a label, 3 with a flap up (the fault times are in the key).
   Rule sentences to use are in `sim/FAULTS.md`. Prefer "label when it enters sealing", "tape before it leaves the view" and "flaps down before tape"; the "at the end of the footage" label rule also flags cartons still upstream (open question below).
   The owner runs the live check on both clips himself. Fix anything general that it shows, and check the fix on the other clip.
   Found on s1 (2026-10-29): SAM finds no "jar" in this render, in video or image mode; "bottle" works. Its label is a plain striped band with no print, and SAM finds it under no name ("label", "white band", "striped band", "label sleeve", whole frame or cropped), so 9 of 10 bottles false-fail. The result now says when nearly every item lacks the part that the name may not match the look. Owner's call: re-render the s1 label with print on it (the sim is frozen), or show labels on s2 only. With "bottle", only 10 of 27 bottles are judged.
   Found on s2: "label" finds the printed labels (0.91); "shipping label" finds none, so the rule says "label". "Every carton has a label when it leaves the view" judged 15 of 18 exits and flagged 1, at 00:47 (the key: 00:45 and 00:58; the 00:58 one was missed). "tape" is not found on a still at all, so the tape rule will need a look.
   Later: the owner adds real stock clips with a visible fault to `data/stock/` (candidates in `data/stock/candidates/`); the truck clip needs the engine to judge an item at its best whole sighting, not its last.
   Done when: both clips show a red box on a real fault in Telegram.
2. **Insights.** Built (how it works is in DECISIONS.md): "What stands out" in the result with a clip button per line, a quiet line for a normal run, and `ask_footage` for questions in chat. Checked live on s1 with `scripts/smoke_insights.py`: the stop is reported as 00:21–00:33 with about 5 bottles lost (the key says 00:20.8–00:33.4), its clip is 1.5 MB, the "who stopped it" question is refused plainly, and insights cost under $0.01. On s2 the first missing label is answered right (00:47 vs 00:45 in the key) and the pace (17 a minute) matches the key; the count is 1 of the key's 2 (the miss above). Not yet checked: the comparison with other recordings on real footage, and the reply in Telegram itself.
   Done when: on a clip with a known fault, the reply gives the right first time and count, and each line links to a clip under 16 MB.
3. **Accuracy table.** Score Rewind against the answer keys of the two sim clips (`ground_truth.json`: which item, what time, what fault): precision, recall, cost, run time. Add a separate row for each real stock clip, hand-labelled, once there are some. One table for the README.
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
