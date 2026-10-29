# Plan

The one place to see what is done, what is next, and in what order. Update it at the end of every work session. If a line would not change what the next session does, delete it.

Deadline: Nov 4, 9:00 PM ET (submit on the evening of Nov 4 IST).

## The demo story

Someone sends a video from their phone (web, Telegram or WhatsApp) and asks in plain words: "anything wrong here?", "when did the belt stop and what did it cost?", "which bottles went out without a label?". Rewind watches, shows each step as it works, and answers in plain words with a clip for every claim. No setup: no lines, no rules to write first. Works on any footage, not only packing lines.

## Pivot (2026-10-30)

We moved from "sentence → strict rule → engine" to "upload, ask, watch it work". Why and how: DECISIONS.md, "Ask the footage". What carries over: the SAM adapter and cache, the cost ledger and caps, the proxy and idle-skip, proof clips, the rule engine (now one tool), the insights claim checker, the Muse adapter with tools.

Muse video check on the jar clip (60 s, contributor model): the belt stop right (00:21–00:33 vs 00:20.8–00:33.4 in the key), 26 items vs 27, 1 of 4 missing labels, the missing cap missed. $0.0025, 43 s. So Muse's whole-video look is the cheap first pass; SAM and crops confirm per-item claims.

## Done

- Project setup: one check command (`uv run poe check`), CI on every PR, `main` protected.
- API checks: SAM 3.1 and Muse Spark called for real; prices and limits are in DECISIONS.md.
- Perception: a video is cut down to its busy parts and sent to SAM once; results are cached. Every paid call is logged and capped.
- Rules engine, proof clips with boxes and a clock, insights with checked claims, Telegram and WhatsApp channels (old flow, with lines and rule drafts).
- Two sim clips with answer keys as test footage: `data/sim/s1_jar_line-seed2` (4 missing labels near 00:08, 00:20, 00:43, 00:48; 1 missing cap; belt stop 00:20.8–00:33.4) and `data/sim/s2_carton_packing-seed2` (3 without tape, 3 without a label, 3 with a flap up). SAM finds no label on s1 under any name; Muse looking at crops is the way now.

## Next, in order

Each phase is one branch and one PR, small enough for one session.

1. **The agent and its tools.** One video is one session (no lines). On upload: make the proxy, then Muse watches the whole video once and returns a scene summary, a timeline of events with times, and 3–5 suggested questions. Then a Muse agent answers any question with tools: `watch` (Muse on the video or a part of it), `find` (SAM tracks a thing), `look` (Muse on crops or frames), `measure` (code: counts, gaps, stops, durations, the rule engine), `clip` (proof clip with boxes). Every step sends a progress event. Every answer cites tool results; code drops any time or number not in them, and marks each claim "measured" or "seen". Per-question budget cap. Check from a script on s1 and s2.
   Done when: on s1, "anything wrong?" names the stop with the right times, and "which jars have no label?" finds at least 3 of 4 with clips.
2. **Web page.** Upload, the video with a timeline of clickable markers, a chat that streams the steps live, clips inline. Warm, neutral, calm colours; clean and minimal. Served by the existing FastAPI app.
   Done when: the owner uploads a clip in the browser, asks two questions and sees each step and each clip.
3. **Telegram and WhatsApp on the new agent.** Send a video: the first look comes back with suggested questions as buttons. Ask anything. Progress is one message edited in place; the answer is one message with the clip. Remove lines, rule drafts and "Looks right" from the flow.
   Done when: the same question gives the same answer on all three channels.
4. **Accuracy table.** Fixed questions on s1 and s2 scored against the answer keys: precision, recall, time error, cost, run time. A real stock clip row if one exists.
5. **Submission.** README with a diagram, a 3-minute demo video, the written answers, credits.

## Only if time is left

- "Keep watching for this": save a question and run it on every new video in the chat.
- Compare two videos ("is today worse than yesterday?").
- An MCP server with the same tools.

If time runs short, cut from the bottom up. Never cut: proof clips, the accuracy table, the web page, one phone channel.

## Submission checklist

- [ ] Public GitHub repo; check it opens logged out.
- [ ] 3-minute demo video on YouTube showing what was built; check it opens logged out.
- [ ] Everyone who worked on it listed on the submission page.
- [ ] Written answers: how we used Meta products, why Meta's models matter here, feedback on them. Feedback notes so far: SAM has no fps setting, a 50-frame minimum and one concept per request; usage is missing on background polls; Muse's prompt cache hit only with `prompt_cache_key`; SAM masks are slow to decode in Python; the video docs do not say how video is billed or sampled.
- [ ] README credits: the Pexels stock clips, and the sim's assets (sim/CREDITS.md).
- [ ] Before judging: run the sample once so later taps are fast and free.

## Open questions for the owner

- Product name: keep "Rewind"?
- Judges try the live product. Is the owner's machine plus a tunnel enough, or do we need an always-on host?

## Owner setup still open

- [ ] WhatsApp keys (names in `.env.example`).
