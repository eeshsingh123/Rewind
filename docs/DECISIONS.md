# Decisions

Why things are the way they are. Only decisions that still shape upcoming work are here; each is short. When one changes, edit it in place.

## Product

**Rules for any camera, told through one packing story.** Plain-English rules run over footage, including footage processed before the rule existed. The pitch: a complaint comes in, a new rule runs over stored footage, and the reply says when it started, how often, and shows the clip. A shelf or dock scene appears only as a short "works here too", once the packing story works on real footage.
Why: "ask your camera anything" builds will be common; judges remember one sharp story.

**Demo on two sim clips; real stock clips come later as the "works on real footage" check.** The demo and the accuracy table start from two rendered clips, each with an exact answer key (`ground_truth.json`): `data/sim/s1_jar_line-seed2` (jars missing a cap or label, plus a belt stop) and `data/sim/s2_carton_packing-seed2` (cartons missing tape or label, or with a flap up). The owner finds real stock clips with a visible fault and adds them when ready. The Blender sim (`sim/`) stays frozen: no new scenes or code.
Why: the stock clips found so far show no clear fault; the sim clips already have known faults and exact answers, and the accuracy table can score against them at once. Real footage still convinces judges more, so it is added as soon as a good clip exists.

**Insights are computed, then worded.** Plain queries over the events find the insights: first and last time a rule failed, bursts and the stops they line up with, downtime and pace, change from earlier in the footage. Each carries its numbers and clips. Only notable ones are sent. Muse only writes the sentence.
Why: every finding must be provable, and an insight shown on every run gets ignored.

**Chat behaviour.** Fixed commands (`/start`, `/rules`, `/clips`, `/help`) make no model call. The first message creates "Line 1"; the sample lives on its own line. A rule is a draft until the user taps "Looks right". A new video is checked against the line's rules by itself. Progress is one checklist message, edited in place. The result is a new message with the worst clip (a green clip when nothing failed). Confirmation is asked only above $0.50, and the cost shown is a floor.

## Architecture

**AI perceives and translates; plain code decides.** SAM 3.1 turns footage into observations once per video and concept, and they are stored. Muse Spark turns sentences into rules. A deterministic Python engine judges the stored observations as often as needed.
Why: re-running a rule on old footage is then cheap, and every decision is testable and repeatable.

**Ports and adapters.** `contracts` ← `core` ← `adapters` ← `app`. A new channel, model or storage is a new adapter, never an edit to the core. import-linter enforces it. Layout is in RULES.md.

**Stack.** Python 3.12 with uv, FastAPI under uvicorn, SQLite now (Postgres later is a URL change), Parquet for observations, ffmpeg and OpenCV.

**Everything runs on the owner's machine.** Blobs on local disk behind `BlobStore`; jobs run in-process. Telegram uses long polling, so it needs no public URL. WhatsApp, upload links and media URLs go through a free `cloudflared` quick tunnel (`scripts/serve_live.py`). Heavy jobs run one at a time (8 GB RAM).
Why: no hosting account or credit needed before the deadline; a host later is new adapters only.

**One agent behind every channel.** Each channel turns its webhooks into one message format and renders replies within its limits. One Muse agent with tools owns the conversation, so every channel gets the same answers.

**Muse model: `muse-spark-1.3-contributor`.** Set by `REWIND_MUSE_MODEL`. About 12–20× cheaper than `muse-spark-1.3`, with every feature we need. Limits: Meta may train on what we send (fine for stock footage), 150 requests per minute.

## Cost

**Prices.** SAM 3.1 video: $0.20 per 1,000 frames; image: $2.50 per 1,000. SAM takes one concept per request, bills every frame sent, has a 50-frame minimum per request and no fps setting. Muse Contributor: $0.10 in, $0.002 cached, $0.20 out per 1M tokens.

**How we keep it cheap.** The frames we send are the bill, so: re-encode to a 720p proxy at the planned fps; skip idle spans (frame differences inside the zones); let the rules decide which concepts and when; track the item continuously but look for parts once per item, with one image call at the moment it is judged; cache by (video, frame, concept, model) so a rerun costs $0. Never one request per item (the 50-frame minimum makes that about 10× dearer).

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
