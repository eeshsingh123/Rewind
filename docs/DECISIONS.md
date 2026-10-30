# Decisions

Why things are the way they are. Only decisions that still shape upcoming work are here; each is short. When one changes, edit it in place.

## Product

**Ask the footage: upload, ask, watch it work (2026-10-30, replaces "rules first").** Any video, any question, no setup. A video is a session; there are no lines and no rules to write first. Muse watches the whole video once on upload and offers a summary, a timeline and questions to try. Each question is answered by a Muse agent that plans its own steps with tools (`watch`, `find`, `look`, `measure`, `clip`), and every step is shown live. Broad scope: packing lines are the main demo, but any footage works.
Why: rules made the product rigid (one kind of question, only items leaving the view) and the setup felt like work. Muse reads a whole video well and cheaply (jar clip: the stop right, $0.0025), so it should plan and reason, not only translate.

**Every claim still has proof.** Answers cite tool results (a time, an item, a clip). Code drops any time or number not in them and marks each claim "measured" (code counted it from SAM tracks) or "seen" (Muse judged it from video or crops). Muse's whole-video look finds where to look; per-item claims are confirmed with SAM tracks and crops before they are stated as counts.
Why: Muse on the whole jar clip got the stop right but found only 1 of 4 missing labels and missed the cap; a close look at each item fixes that.

**Three channels: web, Telegram, WhatsApp.** The web page is where results are easiest to follow (video, timeline markers, live steps). Telegram and WhatsApp show it works from any phone. All three call the same agent and render the same progress events.

**UI: clean, minimal, warm neutral colours.** Calm warm neutrals, one muted accent, plain type, no gradients or loud colours.

**Test footage: two sim clips with answer keys.** `data/sim/s1_jar_line-seed2` and `data/sim/s2_carton_packing-seed2`, each with `ground_truth.json`. Real stock clips are added when one shows a visible fault. The Blender sim (`sim/`) stays frozen.

**Insights: code finds the facts, Muse picks and words them, code checks every claim.** Each check saves a timeline per recording: every item each rule judged, pass or fail, and the stops. A stop is a gap in output longer than 3× the line's own usual gap (at least 5 s); the idle spans say whether the line stood still. From that, code writes a numbered fact sheet: stops and the output they cost, pace, each failure, the other recordings of the line, and patterns with how often chance alone would make them (failures in a row, failures that started or stopped at a moment, every Nth item, failures right after a restart). Notable means chance would make it less than 1 time in 20. Muse is called only when there is a notable fact. It gets the sheet plus one picture of cropped tiles (failed items boxed, the stop) and returns at most 3 insights, each citing its facts and a clip. Code drops any insight that cites an unknown fact or states a number or time that is not in its facts. A normal run gets one quiet line from code, and no call. Questions in chat (`ask_footage`) use the same sheet: unanswerable ones say why and suggest a rule. Answers are cached by their inputs, so the same question never costs twice.
Why: every finding must be provable, and an insight shown on every run gets ignored. Muse adds what the numbers cannot: picking what matters, linking facts, and saying what a failure looks like.

**Chat behaviour.** Fixed commands (`/start`, `/help`) make no model call. A video starts a session and gets the first look with suggested questions as buttons. Any message after that is a question about the latest video. Progress is one message, edited in place. The answer is one message with its clip. Confirmation is asked only above $0.50.

## Architecture

**Muse plans and reasons; SAM tracks; code measures and checks.** Muse watches video, looks at crops and decides which tools to call. SAM 3.1 tracks things once per video and concept, stored and cached. Code counts and times from those tracks, draws the proof, and checks every claim in the answer. The rule engine is one `measure` tool.
Why: Muse is cheap and good at the whole picture; SAM and code make per-item counts and times exact and repeatable.

**Muse video.** `input_video` with a `file_id` from the Files API (MP4 only, set an expiry). Muse reads the visual sequence and any audio. The docs do not say how video is sampled or billed; on the 60 s jar clip it was 15.7k input tokens and 43 s.

**Ports and adapters.** `contracts` ← `core` ← `adapters` ← `app`. A new channel, model or storage is a new adapter, never an edit to the core. import-linter enforces it. Layout is in RULES.md.

**Stack.** Python 3.12 with uv, FastAPI under uvicorn, SQLite now (Postgres later is a URL change), Parquet for observations, ffmpeg and OpenCV.

**Everything runs on the owner's machine.** Blobs on local disk behind `BlobStore`; jobs run in-process. Telegram uses long polling, so it needs no public URL. WhatsApp, upload links and media URLs go through a free `cloudflared` quick tunnel (`scripts/serve_live.py`). Heavy jobs run one at a time (8 GB RAM).
Why: no hosting account or credit needed before the deadline; a host later is new adapters only.

**One agent behind every channel.** Each channel turns its webhooks into one message format and renders replies within its limits. One Muse agent with tools owns the conversation, so every channel gets the same answers.

**Muse model: always `muse-spark-1.3-contributor`.** Set by `REWIND_MUSE_MODEL`; also for tests and one-off checks. About 12–20× cheaper than `muse-spark-1.3`, with every feature we need, video included. Limits: Meta may train on what we send (fine for stock footage), 150 requests per minute.

## Cost

**Prices.** SAM 3.1 video: $0.20 per 1,000 frames; image: $2.50 per 1,000. SAM takes one concept per request, bills every frame sent, has a 50-frame minimum per request and no fps setting. Muse Contributor: $0.10 in, $0.002 cached, $0.20 out per 1M tokens.

**Budget for experiments: $150 in total.** Use Muse freely (a whole-video look costs well under a cent per minute of footage). SAM is the costly call; the agent calls it only when a question needs per-item counts or times.

**How we keep SAM cheap.** The frames we send are the bill, so: re-encode to a 720p proxy at the planned fps; skip idle spans (frame differences inside the zones); let the rules decide which concepts and when; track the item continuously but look for parts once per item, with one image call at the moment it is judged; cache by (video, frame, concept, model) so a rerun costs $0. Never one request per item (the 50-frame minimum makes that about 10× dearer).

**Guardrails.** Every paid call goes through the ledger and the budget guard. A cost estimate comes before every check. Each workspace has a daily cap; a global kill switch stops all paid calls at $120 (never above $140). A failed paid call still counts toward the caps. Webhooks are answered 200 at once and de-duplicated by message id. The agent has caps on tool calls and tokens per turn.
Why: no runaway spend; webhook retries and agent loops are how costs usually blow up.

## How rules are judged

- `at_checkpoint`: the part was seen within the 1.0 s before the checkpoint. `ever_before`: seen any time from the item's first sighting to the checkpoint.
- "Leaves the view" is judged only when a track ends at the frame border more than 1 s before the footage ends. Items still in view, lost mid-frame, never seen whole, or not moving are not judged. Parts are checked in the last second the item was whole in frame, not the sliver at the border.
- "At the end of the footage" judges every item as last seen. A sentence that names no moment means "leaves the view".
- `count_at_least` counts parts in one frame. `before(a, b)` compares first sightings.
- One violation per item per checkpoint, with the reasons joined. Rates are judged per whole minute.
- SAM's object ids are reused across items, so item identity is our own tracking. SAM follows at most 16 objects per request; when full, the rest of the window is sent again from that frame.
- Boxes and zones are fractions of the frame (0..1), so they survive any proxy resolution.

## Accuracy

**The accuracy table is the proof.** Per-item precision and recall (a match is the same item within ±2 s), cost and run time, on hand-labelled real clips. Never cut.
