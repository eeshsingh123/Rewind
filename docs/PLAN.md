# Plan

The one place to see what is done, what is next, and in what order. Update it at the end of every work session. If a line would not change what the next session does, delete it.

Deadline: Nov 4, 9:00 PM ET (submit on the evening of Nov 4 IST).

## The demo story

Someone sends a photo or a video from their phone (web, Telegram or WhatsApp) and asks in plain words: "anything wrong here?", "when did the belt stop and what did it cost?", "does the top row of the shelf have dairy products?". Rewind looks, shows each step as it works, and answers in plain words with proof for every claim: a clip, or the photo with boxes drawn. No setup: no lines, no rules to write first. Works on anything a camera can see, from factory lines to dark stores, shops and supermarkets.

## Pivot (2026-10-30)

We moved from "sentence → strict rule → engine" to "upload, ask, watch it work". Why and how: DECISIONS.md, "Ask the footage". What carries over: the SAM adapter and cache, the cost ledger and caps, the proxy and idle-skip, proof clips, the rule engine (now one tool), the insights claim checker, the Muse adapter with tools.

Muse video check on the jar clip (60 s, contributor model): the belt stop right (00:21–00:33 vs 00:20.8–00:33.4 in the key), 26 items vs 27, 1 of 4 missing labels, the missing cap missed. $0.0025, 43 s. So Muse's whole-video look is the cheap first pass; SAM and crops confirm per-item claims.

## Done

- Project setup: one check command (`uv run poe check`), CI on every PR, `main` protected.
- API checks: SAM 3.1 and Muse Spark called for real; prices and limits are in DECISIONS.md.
- Perception: a video is cut down to its busy parts and sent to SAM once; results are cached. Every paid call is logged and capped.
- Rules engine, proof clips with boxes and a clock, insights with checked claims, Telegram and WhatsApp channels (old flow, with lines and rule drafts).
- Two sim clips with answer keys as test footage: `data/sim/s1_jar_line-seed2` (4 missing labels near 00:08, 00:20, 00:43, 00:48; 1 missing cap; belt stop 00:20.8–00:33.4) and `data/sim/s2_carton_packing-seed2` (3 without tape, 3 without a label, 3 with a flap up). SAM finds no label on s1 under any name; Muse looking at crops is the way now.

- Upload and ask, the agent and its tools (**not yet checked live**). One photo or video is one session. On upload Muse goes over it once and returns a summary, what stands out and 3–5 questions; the result is cached. A Muse agent answers each question with `watch` (video only), `find`, `look`, `measure` and `area`. Every step sends a progress event. Code drops any claim whose numbers or times are not in its cited evidence, marks each kept claim "measured" or "seen", and attaches the proof. Each question has a $0.50 budget and a cap on steps and tokens. A photo has no times: SAM finds items with one image call, items are named by place ("the 2nd box from the left"), and the proof is the photo with boxes. An area is marked by Muse and confirmed by SAM, and code limits counts to it.

## Next, in order

Each phase is one branch and one PR, small enough for one session.

1. **Live check of the agent.** Nothing in it has run against the APIs yet. Run `scripts/smoke_ask.py` on the s1 clip, then on a photo with a question about one area.
   Done when: on s1, "anything wrong?" names the stop with the right times, and "which jars have no label?" finds at least 3 of 4 with clips; on the photo, an area is marked and confirmed and its count is right.
2. **Web page.** Upload a photo or a video; a video gets a timeline of clickable markers. A chat streams the steps live, with proof inline. Warm, neutral, calm colours; clean and minimal. Served by the existing FastAPI app. Also:
   - Ready samples: 2–3 clips a visitor taps to get a cached answer at once, at no cost.
   - Each answer ends with a quiet cost line ("cost $0.003").
   - Point to track: drag a box on a frame and ask "follow this"; SAM tracks it from the box prompt.
   - Every session has its own link, so a chat can point to it.
   Done when: the owner uploads a clip in the browser, asks two questions and sees each step and each clip; a sample answers at once; a dragged box is tracked.
3. **Telegram and WhatsApp on the new agent.** Send a photo or a video: the first look comes back with suggested questions as buttons. Ask anything. Progress is one message edited in place; the answer is one message with the clip. Remove lines, rule drafts and "Looks right" from the flow. Also:
   - Voice notes in any language: Meta's speech-to-text turns the note into the question, and the answer comes back in the language asked.
   - Every answer ends with "Open the full timeline", linking to that session's web page.
   Done when: the same question gives the same answer on all three channels, and a Hindi voice note gets a Hindi answer.
4. **Accuracy table.** Fixed questions on s1 and s2 scored against the answer keys: precision, recall, time error, cost, run time. A row for the owner's real clip (below), hand-labelled.
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

## Owner setup still open

- [ ] WhatsApp keys (names in `.env.example`).
- [ ] Film one real 1–2 minute phone clip with a known fault (for example, packing boxes on a table with one left untaped) and note the fault times. Used in the demo and the accuracy table.
- [ ] A real shelf or store photo or video, to check changes on a second kind of footage.
- [ ] Decide hosting before phase 3: judges try it live, and a tunnel to the laptop works only while it is on.
