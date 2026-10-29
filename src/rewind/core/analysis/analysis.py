import dataclasses
import hashlib
import logging
import math
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Protocol

from pydantic import JsonValue, TypeAdapter

from rewind.contracts import (
    Button,
    ChannelCapabilities,
    Charge,
    CostEstimate,
    Footage,
    Line,
    Observation,
    ReplyMedia,
    Rule,
    RuleSpec,
    TimeSpan,
    Violation,
    Zone,
)
from rewind.core.analysis.insights import Answer, Found, Insights, SavedInsight
from rewind.core.analysis.parts import (
    PartLooker,
    StillGrabber,
    checkpoint_parts,
    still_times,
    tracking_spec,
)
from rewind.core.analysis.progress import Checklist, Notifier, Step, about_time
from rewind.core.analysis.proof import (
    ClipMaker,
    Mark,
    ProofMarks,
    marks_key,
    passed_marks,
    proof_marks,
)
from rewind.core.analysis.replies import (
    RuleOutcome,
    clip_caption,
    duration_text,
    join_words,
    plural,
    rule_result,
    video_length,
)
from rewind.core.analysis.timeline import Judged, judged_items, merged_timeline
from rewind.core.analysis.tracks import naive_tracks
from rewind.core.budget import BudgetExceededError, BudgetScope
from rewind.core.perception import (
    CacheKey,
    Ingest,
    ObservationStore,
    PerceptionRunner,
    PlanInput,
    Planner,
    frame_index,
)
from rewind.core.ports import (
    BlobStore,
    ConceptPrompt,
    JobPayload,
    Repository,
    Segmenter,
    SegmenterError,
    SegmentRequest,
    VideoTools,
    VideoToolsError,
)
from rewind.core.rules import (
    Check,
    Item,
    Keyframe,
    RuleCompileError,
    RuleCompiler,
    RuleError,
    Scene,
    build_items,
    checks,
    evaluate,
    find_zone,
    zone_keys,
)

logger = logging.getLogger(__name__)

INGEST_JOB = "ingest"
ANALYZE_JOB = "analyze"
_SPANS = TypeAdapter(tuple[TimeSpan, ...])
# Estimates price frames only; the prompt is never sent.
_NO_PROMPT = ConceptPrompt("estimate")
# How long a check takes, roughly: a fixed start, then each frame SAM still has to see (its
# requests, mask decoding, and the stills of the items it finds; measured on the stock conveyor
# clip, 2026-10-29), and each frame already cached. The cost shown is a floor: stills per item
# and extra requests on a busy line are only known as the check runs.
START_S = 15.0
PAID_FRAME_S = 6.5
CACHED_FRAME_S = 0.2
# The sample's line is per workspace and never the default line, so a person's own videos are
# not checked against the sample rule.
_SAMPLE_LINE = "-sample"


class Segmenters(Protocol):
    # Callers only ever get segmenters wrapped in the budget guard.
    def video(self, scope: BudgetScope) -> Segmenter: ...

    def image(self, scope: BudgetScope) -> Segmenter: ...


@dataclass(frozen=True, slots=True)
class AnalysisSettings:
    fps: float
    strategy: str = "dense_track"


@dataclass(frozen=True, slots=True)
class FootageEstimate:
    footage: Footage
    frames: int
    cached_frames: int
    usd: Decimal


@dataclass(frozen=True, slots=True)
class Estimate:
    footage: tuple[FootageEstimate, ...]
    # Areas the rules name that are not drawn: each is found once per line from one still.
    zones_to_find: tuple[str, ...]
    zones_usd: Decimal
    # Parts looked for on one still per item; how many items there are is known only after.
    parts_on_stills: tuple[str, ...]
    per_item_usd: Decimal
    seconds: float

    @property
    def total_usd(self) -> Decimal:
        return sum((f.usd for f in self.footage), self.zones_usd)


@dataclass(frozen=True, slots=True)
class Clip:
    # Empty when the chat cannot take the clip.
    media: tuple[ReplyMedia, ...]
    caption: str


@dataclass(frozen=True, slots=True)
class Read:
    line: Line
    footage: Footage
    # What arrived and where something moves, in the person's words.
    summary: str
    # What the person can do next with the line's rules as they are.
    next_step: str


@dataclass(frozen=True, slots=True)
class Checked:
    line: Line
    outcomes: tuple[RuleOutcome, ...]
    cost: Decimal
    skipped: tuple[str, ...]
    # What to show when nothing failed: every judged item around the first one, in green.
    passed: tuple[TimeSpan, ProofMarks] | None
    found: Found


@dataclass(frozen=True, slots=True)
class CheckRequest:
    # How the check is named in its checklist: "the sample", "your video".
    title: str | None = None
    # Said above the checklist, such as "Saved as rule 1.".
    lead: str = ""
    estimate_s: float = 0.0
    estimate_usd: Decimal = Decimal(0)


def footage_name(footage: Footage) -> str:
    return PurePosixPath(footage.original_key).name


def sample_line_id(workspace_id: str) -> str:
    return f"{workspace_id}{_SAMPLE_LINE}"


def is_sample_line(line: Line) -> bool:
    return line.id == sample_line_id(line.workspace_id)


def check_steps(footage: Footage, rules: Sequence[Rule]) -> list[Step]:
    entities = sorted({rule.spec.entity.phrase for rule in rules})
    parts = sorted({phrase for rule in rules for phrase in checkpoint_parts(rule.spec)})
    steps = [
        Step("received", "Video received", "Video received", "done"),
        Step("read", "Reading the video", f"Video read ({video_length(footage.duration_s)})",
             "done"),
        Step("find", f"Finding {join_words([plural(e, 2) for e in entities])}"),
    ]  # fmt: skip
    if parts:
        steps.append(Step("parts", f"Looking for {join_words([plural(p, 2) for p in parts])}"))
    steps.append(Step("rules", f"Checking {len(rules)} {plural('rule', len(rules))}"))
    steps.append(Step("insights", "Looking for patterns"))
    steps.append(Step("clip", "Making the clip"))
    return steps


def check_header(request: CheckRequest, what: str) -> str:
    return (
        f"{request.lead}Checking {what} · {about_time(request.estimate_s)} · "
        f"from ${request.estimate_usd:.2f}"
    )


def check_request(payload: JobPayload) -> CheckRequest:
    title = payload.get("title")
    return CheckRequest(
        title=str(title) if title else None,
        lead=str(payload.get("lead") or ""),
        estimate_s=float(str(payload.get("estimate_s") or 0)),
        estimate_usd=Decimal(str(payload.get("estimate_usd") or 0)),
    )


def check_payload(request: CheckRequest) -> JobPayload:
    payload: JobPayload = {
        "lead": request.lead,
        "estimate_s": request.estimate_s,
        "estimate_usd": str(request.estimate_usd),
    }
    if request.title:
        payload["title"] = request.title
    return payload


def check_title(request: CheckRequest, footage: Footage, index: int, count: int) -> str:
    if count > 1:
        return f"*{footage_name(footage)}* ({index} of {count})"
    return request.title or "your video"


class Analysis:
    def __init__(
        self,
        repository: Repository,
        blobs: BlobStore,
        ingest: Ingest,
        planner: Planner,
        runner: PerceptionRunner,
        observations: ObservationStore,
        segmenters: Segmenters,
        compiler_for: Callable[[str], RuleCompiler],
        video: VideoTools,
        stills: StillGrabber,
        clips: ClipMaker,
        insights: Insights,
        notifier: Notifier,
        capabilities: Mapping[str, ChannelCapabilities],
        settings: AnalysisSettings,
    ) -> None:
        self._repository = repository
        self._blobs = blobs
        self._ingest = ingest
        self._planner = planner
        self._runner = runner
        self._observations = observations
        self._segmenters = segmenters
        self._compiler_for = compiler_for
        self._video = video
        self._stills = stills
        self._clips = clips
        self._insights = insights
        self._notifier = notifier
        self._capabilities = capabilities
        self._settings = settings
        self._parts = PartLooker(blobs, observations, stills)

    # Reads a received video and saves it as footage; None when its line is gone.
    async def read(self, payload: JobPayload) -> Read | None:
        line = await self._repository.get_line(str(payload["line_id"]))
        if line is None:
            return None
        drawn = tuple(zone for zone in line.zones if zone.source == "drawn")
        ingested = await self._ingest.ingest(
            str(payload["footage_id"]), line.id, str(payload["original_key"]), drawn
        )
        footage = ingested.footage
        await self._blobs.write(_spans_key(footage.id), _SPANS.dump_json(ingested.active_spans))
        await self._repository.save_footage(footage)
        moving = sum(span.end - span.start for span in ingested.active_spans)
        summary = f"✓ Got your video ({video_length(footage.duration_s)})."
        if moving >= footage.duration_s - 1:
            summary += " Something moves for the whole clip."
        elif moving == 0:
            summary += " Nothing moves in it, so there is nothing to check."
        else:
            summary += f" Something moves for {duration_text(moving)}; I'll only check that."
        next_step = (
            "Now tell me one rule in plain words, for example: "
            '"every box has a label when it leaves the view".'
        )
        return Read(line, footage, summary, next_step)

    async def estimate(
        self, workspace_id: str, line: Line, footage: Sequence[Footage], rules: Sequence[Rule]
    ) -> Estimate:
        specs = tuple(rule.spec for rule in rules)
        segmenter = self._segmenters.video(BudgetScope(workspace_id))
        estimates = [await self._estimate_footage(footage_item, specs, segmenter)
                     for footage_item in footage]  # fmt: skip
        image = self._segmenters.image(BudgetScope(workspace_id))
        one_image = Decimal(0)
        if footage:
            request = SegmentRequest(footage[0].id, "still.jpg", 1, 1, (0.0,), _NO_PROMPT)
            one_image = CostEstimate(charges=image.estimate(request)).total_usd
        missing = _missing_zones(line, specs)
        on_stills = tuple(sorted({phrase for spec in specs for phrase in checkpoint_parts(spec)}))
        paid = sum(e.frames - e.cached_frames for e in estimates)
        cached = sum(e.cached_frames for e in estimates)
        seconds = START_S * len(estimates) + PAID_FRAME_S * paid + CACHED_FRAME_S * cached
        return Estimate(
            tuple(estimates), missing, one_image * len(missing), on_stills,
            one_image * len(on_stills), seconds,
        )  # fmt: skip

    async def handle_analyze(self, payload: JobPayload) -> None:
        workspace_id = str(payload["workspace_id"])
        channel, chat_id = str(payload["channel"]), str(payload["chat_id"])
        job_id = str(payload["job_id"])
        key = str(payload.get("progress_key") or job_id)
        request = check_request(payload)
        scope = BudgetScope(workspace_id, job_id)
        line = await self._repository.get_line(str(payload["line_id"]))
        rule_ids = _strings(payload["rule_ids"])
        footage_ids = _strings(payload["footage_ids"])
        if line is None:
            return
        rules = [rule for rule in await self._repository.rules(line.id) if rule.id in rule_ids]
        numbers = await self.rule_numbers(line.id)
        started = time.monotonic()
        spent = Decimal(0)
        checked: list[tuple[Footage, Checked]] = []
        problems: list[str] = []
        checklist: Checklist | None = None
        what = ""
        for index, footage_id in enumerate(footage_ids, 1):
            footage = await self._repository.get_footage(footage_id)
            if footage is None:
                continue
            what = check_title(request, footage, index, len(footage_ids))
            checklist = Checklist(
                self._notifier, channel, chat_id, key, check_header(request, what),
                check_steps(footage, rules),
            )  # fmt: skip
            await checklist.show()
            try:
                result = await self._check(line, footage, rules, numbers, scope, checklist)
            except BudgetExceededError as error:
                problems.append(f"I stopped before *{footage_name(footage)}*: {error}.")
                break
            except (SegmenterError, VideoToolsError, RuleCompileError):
                logger.exception("analysis of %s failed", footage.id)
                problems.append(
                    f"*{footage_name(footage)}*: something went wrong while checking it. Try again."
                )
                continue
            line = result.line
            spent += result.cost
            checked.append((footage, result))
        media = await self._result_clip(line, channel, checked, checklist)
        if checklist is not None:
            elapsed = duration_text(time.monotonic() - started)
            verb = "Checked" if checked else "Couldn't check"
            checklist.header = f"{request.lead}{verb} {what} in {elapsed} · ${spent:.2f}"
            await checklist.show()
        text = _result_text(checked, media, problems)
        buttons = await self._result_buttons(line, checked, footage_ids)
        if media is not None:
            media = media.model_copy(update={"caption": text})
            await self._notifier.notify(channel, chat_id, "", media=(media,), buttons=buttons)
            return
        await self._notifier.notify(channel, chat_id, text, buttons=buttons)

    async def clip(
        self, footage: Footage, violation: Violation, number: int, count: int, max_bytes: int
    ) -> Clip:
        numbers = await self.rule_numbers(footage.line_id)
        caption = clip_caption(violation, number, count, numbers.get(violation.rule_id))
        if max_bytes <= 0:
            return Clip((), caption)
        stored = marks_key(violation.id)
        marks = (
            _MARKS.validate_json(await self._blobs.read(stored))
            if await self._blobs.exists(stored)
            else ProofMarks(marks=())
        )
        span = TimeSpan(
            start=max(0.0, violation.clip.start), end=min(footage.duration_s, violation.clip.end)
        )
        key = await self._render(footage, span, marks, violation.id)
        if (await self._blobs.local_path(key)).stat().st_size > max_bytes:
            return Clip((), caption + "\n(The clip is too big to send on this chat.)")
        return Clip((ReplyMedia(kind="video", blob_key=key, caption=caption),), caption)

    async def ask(
        self, workspace_id: str, footage: Footage, question: str, window: TimeSpan | None
    ) -> Answer | None:
        line = await self._repository.get_line(footage.line_id)
        if line is None:
            return None
        numbers = await self.rule_numbers(line.id)
        return await self._insights.ask(workspace_id, line, footage, numbers, question, window)

    async def insight(self, footage: Footage, insight_id: str) -> SavedInsight | None:
        return (await self._insights.saved(footage.id)).get(insight_id)

    async def insight_clip(self, footage: Footage, insight: SavedInsight, max_bytes: int) -> Clip:
        if max_bytes <= 0 or insight.clip is None:
            return Clip((), insight.text)
        marks: dict[tuple[float, float, float], Mark] = {}
        for violation_id in insight.violation_ids:
            stored = marks_key(violation_id)
            if not await self._blobs.exists(stored):
                continue
            for mark in _MARKS.validate_json(await self._blobs.read(stored)).marks:
                first = mark.path[0]
                key = (first.t, first.box.x0, first.box.y0)
                # The same item is drawn once, flagged when any violation flags it.
                if key not in marks or mark.kind == "flagged":
                    marks[key] = mark
        span = TimeSpan(start=max(0.0, insight.clip.start),
                        end=min(footage.duration_s, insight.clip.end))  # fmt: skip
        key = await self._render(footage, span, ProofMarks(marks=tuple(marks.values())),
                                 f"insight-{insight.id}")  # fmt: skip
        if (await self._blobs.local_path(key)).stat().st_size > max_bytes:
            return Clip((), insight.text + "\n(The clip is too big to send on this chat.)")
        return Clip((ReplyMedia(kind="video", blob_key=key, caption=insight.text),), insight.text)

    # The clip that goes with the result: the first failure of the rule that failed most, or,
    # when nothing failed, the items that passed.
    async def _result_clip(
        self,
        line: Line,
        channel: str,
        checked: Sequence[tuple[Footage, Checked]],
        checklist: Checklist | None,
    ) -> ReplyMedia | None:
        max_bytes = self._capabilities[channel].max_media_bytes
        worst = _worst(checked)
        passed = next(((f, c.passed) for f, c in checked if c.passed is not None), None)
        if max_bytes <= 0 or (worst is None and passed is None):
            if checklist is not None:
                await checklist.skip("clip", "No clip to make")
            return None
        if checklist is not None:
            await checklist.start("clip")
        media: ReplyMedia | None = None
        try:
            if worst is not None:
                footage, violation = worst
                found = await self._repository.violations(footage.id)
                number = [v.id for v in found].index(violation.id) + 1
                clip = await self.clip(footage, violation, number, len(found), max_bytes)
                media = clip.media[0] if clip.media else None
            elif passed is not None:
                footage, (span, marks) = passed
                key = await self._render(footage, span, marks, f"{footage.id}-passed")
                if (await self._blobs.local_path(key)).stat().st_size <= max_bytes:
                    media = ReplyMedia(kind="video", blob_key=key)
        except VideoToolsError:
            logger.exception("could not make the result clip for line %s", line.id)
        if checklist is not None:
            if media is None:
                await checklist.skip("clip", "Couldn't make the clip")
            else:
                await checklist.finish("clip", "Made the clip")
        return media

    async def _result_buttons(
        self, line: Line, checked: Sequence[tuple[Footage, Checked]], footage_ids: Sequence[str]
    ) -> tuple[Button, ...]:
        buttons: list[Button] = [
            Button(label=f"▶ {insight.label}", value=f"/insight {footage.id} {insight.id}")
            for footage, result in checked
            for insight in result.found.insights
            if insight.clip is not None
        ]
        for footage, result in checked:
            count = sum(len(o.violations) for o in result.outcomes)
            if count > 1:
                label = f"Show all clips ({count})"
                buttons.append(Button(label=label, value=f"/clip {footage.id} 1"))
                break
        footage = await self._repository.line_footage(line.id)
        older = [f for f in footage if f.id not in footage_ids]
        if is_sample_line(line):
            buttons.append(Button(label="Send my own video", value="/send_video"))
        elif older:
            buttons.append(Button(label="Check older footage", value=f"/older {line.id}"))
        else:
            buttons.append(Button(label="Send another video", value="/send_video"))
        buttons.append(Button(label="Add a rule", value="/add_rule"))
        return tuple(buttons)

    # Named by what is drawn, so a rerun that changes the boxes renders a new clip.
    async def _render(self, footage: Footage, span: TimeSpan, marks: ProofMarks, name: str) -> str:
        drawn = hashlib.sha256(_MARKS.dump_json(marks)).hexdigest()[:12]
        key = f"proof/{name}-{drawn}.mp4"
        if await self._blobs.exists(key):
            return key
        source = await self._blobs.local_path(footage.original_key)
        with tempfile.TemporaryDirectory() as workdir:
            destination = Path(workdir) / "clip.mp4"
            if marks.marks:
                await self._clips.make_clip(source, destination, span, marks.marks)
            else:
                await self._video.cut_clip(source, destination, span)
            await self._blobs.write_file(key, destination)
        return key

    async def rule_numbers(self, line_id: str) -> dict[str, int]:
        rules = [rule for rule in await self._repository.rules(line_id) if rule.active]
        return {rule.id: index for index, rule in enumerate(rules, 1)}

    async def _check(
        self,
        line: Line,
        footage: Footage,
        rules: Sequence[Rule],
        numbers: Mapping[str, int],
        scope: BudgetScope,
        checklist: Checklist,
    ) -> Checked:
        spans = await self.active_spans(footage)
        tracked = tuple(tracking_spec(rule.spec) for rule in rules)
        plan = self._planner.plan(
            self._settings.strategy, PlanInput(footage, spans, tracked, self._settings.fps)
        )
        await checklist.start("find")

        async def finding(done: int, total: int) -> None:
            await checklist.count("find", done, total)

        result = await self._runner.run(footage, plan, self._segmenters.video(scope), finding)
        charges = list(result.charges)
        line, zone_charges = await self._find_zones(line, footage, spans, rules, scope)
        charges += zone_charges
        tracks = naive_tracks(footage.id, result.observations, max(1.0, 3 / self._settings.fps))
        await checklist.finish("find", _found_text(rules, [t.concept for t in tracks]))
        scene = Scene(footage.id, footage.duration_s, tracks, result.observations, line.zones)
        judged: dict[str, tuple[Check, ...]] = {}
        skipped: list[str] = []
        times: defaultdict[str, set[float]] = defaultdict(set)
        for rule in rules:
            try:
                judged[rule.id] = checks(rule.spec, scene)
            except RuleError as error:
                skipped.append(f"I couldn't check rule {numbers.get(rule.id, '?')}: {error}.")
                continue
            for concept, wanted in still_times(rule.spec, judged[rule.id]).items():
                times[concept] |= wanted
        if checklist.has("parts"):
            await checklist.start("parts")

        async def looking(done: int, total: int) -> None:
            await checklist.count("parts", done, total)

        looks = await self._parts.look(footage, times, self._segmenters.image(scope), looking)
        if checklist.has("parts"):
            items = len({c.track_id for found in judged.values() for c in found})
            await checklist.finish("parts", _looked_text(rules, items))
        charges += looks.charges
        scene = dataclasses.replace(scene, observations=scene.observations + looks.observations)
        await checklist.start("rules")
        outcomes: list[RuleOutcome] = []
        passed: tuple[TimeSpan, ProofMarks] | None = None
        every: dict[str, tuple[Judged, ...]] = {}
        for rule in rules:
            if rule.id not in judged:
                continue
            found = evaluate(rule.id, rule.spec, scene)
            items = await self._save_marks(rule.spec, scene, judged[rule.id], found)
            number = numbers.get(rule.id, 0)
            outcomes.append(RuleOutcome(number, rule, len(judged[rule.id]), found))
            every[rule.id] = judged_items(rule.id, judged[rule.id], found)
            if passed is None:
                passed = passed_marks(items, judged[rule.id], footage.duration_s)
        violations = [v for outcome in outcomes for v in outcome.violations]
        await self._repository.replace_violations(footage.id, list(judged), violations)
        await checklist.finish("rules", f"Checked {len(rules)} {plural('rule', len(rules))}")
        previous = await self._insights.timeline(footage.id)
        entities = {rule.spec.entity.phrase for rule in rules}
        paths = [
            [(o.t, (o.box.x0 + o.box.x1) / 2, (o.box.y0 + o.box.y1) / 2)
             for o in sorted(track.observations, key=lambda o: o.t)]
            for track in tracks
            if track.concept in entities
        ]  # fmt: skip
        timeline = merged_timeline(previous, footage.id, footage.duration_s, every, spans, paths)
        await self._insights.save_timeline(timeline)
        await checklist.start("insights")
        insights = await self._insights.find(scope.workspace_id, line, footage, numbers)
        charges += insights.charges
        noticed = len(insights.insights)
        await checklist.finish(
            "insights",
            f"Found {noticed} {plural('thing', noticed)} worth a look" if noticed
            else "Nothing unusual",
        )  # fmt: skip
        cost = CostEstimate(charges=tuple(charges)).total_usd
        return Checked(line, tuple(outcomes), cost, tuple(skipped), passed, insights)

    # What each proof clip draws is saved now, while the tracks are at hand, so a clip asked
    # for later shows exactly the boxes this check decided on.
    async def _save_marks(
        self,
        spec: RuleSpec,
        scene: Scene,
        judged: Sequence[Check],
        found: Sequence[Violation],
    ) -> dict[str, Item]:
        items = {
            item.track.track_id: item
            for item in build_items(spec, scene.tracks, scene.observations, {})
        }
        failed = {v.track_id: v for v in found if v.track_id is not None}
        for violation in found:
            if violation.track_id is None:
                continue
            marks = proof_marks(violation, items, judged, failed)
            await self._blobs.write(marks_key(violation.id), _MARKS.dump_json(marks))
        return items

    # Areas a rule names but nobody drew are found by text on one still, then kept on the line
    # so the next recording of the same camera needs no call.
    async def _find_zones(
        self,
        line: Line,
        footage: Footage,
        spans: Sequence[TimeSpan],
        rules: Sequence[Rule],
        scope: BudgetScope,
    ) -> tuple[Line, tuple[Charge, ...]]:
        missing = _missing_zones(line, [rule.spec for rule in rules])
        if not missing:
            return line, ()
        phrases: dict[str, str] = {}
        compiler = self._compiler_for(scope.workspace_id)
        for rule in rules:
            # The compile cache makes this free: the same text never compiles twice.
            compiled = await compiler.compile(rule.text)
            phrases.update(compiled.zone_phrases)
        t = (spans[0].start + spans[0].end) / 2 if spans else footage.duration_s / 2
        image_key = f"stills/{footage.id}/{t:.2f}.jpg"
        if not await self._blobs.exists(image_key):
            source = await self._blobs.local_path(footage.original_key)
            with tempfile.TemporaryDirectory() as workdir:
                destination = Path(workdir) / "still.jpg"
                await self._stills.still(source, t, destination)
                await self._blobs.write_file(image_key, destination)
        keyframe = Keyframe(footage.id, image_key, footage.width, footage.height, t)
        segmenter = self._segmenters.image(scope)
        charges: list[Charge] = []
        found: list[Zone] = []
        for key in missing:
            result = await find_zone(segmenter, key, phrases.get(key, key.replace("_", " ")),
                                     keyframe)  # fmt: skip
            charges += result.charges
            if result.zone is not None:
                found.append(result.zone)
        if found:
            line = line.model_copy(update={"zones": (*line.zones, *found)})
            await self._repository.save_line(line)
        return line, tuple(charges)

    async def active_spans(self, footage: Footage) -> tuple[TimeSpan, ...]:
        key = _spans_key(footage.id)
        if await self._blobs.exists(key):
            return _SPANS.validate_json(await self._blobs.read(key))
        return (TimeSpan(start=0, end=footage.duration_s),)

    async def _estimate_footage(
        self, footage: Footage, specs: tuple[RuleSpec, ...], segmenter: Segmenter
    ) -> FootageEstimate:
        spans = await self.active_spans(footage)
        tracked = tuple(tracking_spec(spec) for spec in specs)
        plan = self._planner.plan(
            self._settings.strategy, PlanInput(footage, spans, tracked, self._settings.fps)
        )
        frames = cached = 0
        charges: list[Charge] = []
        concepts = {c for window in plan.windows for c in window.concepts}
        for concept in sorted(concepts):
            key = CacheKey(footage.content_hash, concept, segmenter.model_version)
            stored = (await self._observations.load(key, footage.id)).frames
            times = [t for w in plan.windows if concept in w.concepts
                     for t in _sample_times(w.span, w.fps)]  # fmt: skip
            uncached = [t for t in times if not _cached(stored, t, footage.fps, self._settings.fps)]
            frames += len(times)
            cached += len(times) - len(uncached)
            for start in range(0, len(uncached), segmenter.max_frames):
                chunk = tuple(uncached[start : start + segmenter.max_frames])
                request = SegmentRequest(footage.id, "estimate.mp4", 1, 1, chunk, _NO_PROMPT)
                charges += segmenter.estimate(request)
        usd = CostEstimate(charges=tuple(charges)).total_usd
        return FootageEstimate(footage, frames, cached, usd)


_MARKS = TypeAdapter(ProofMarks)


def _result_text(
    checked: Sequence[tuple[Footage, Checked]], media: ReplyMedia | None, problems: Sequence[str]
) -> str:
    worst = _worst(checked)
    clipped = worst[1] if worst is not None and media is not None else None
    sections: list[str] = []
    for footage, result in checked:
        blocks = [rule_result(o, clipped if clipped in o.violations else None)
                  for o in result.outcomes]  # fmt: skip
        blocks += result.skipped
        if result.found.insights:
            lines = [f"• {insight.text}" for insight in result.found.insights]
            blocks.append("\n".join(["*What stands out*", *lines]))
        elif result.found.quiet:
            blocks.append(result.found.quiet)
        if len(checked) > 1:
            blocks.insert(0, f"*{footage_name(footage)}* ({video_length(footage.duration_s)})")
        sections.append("\n\n".join(blocks))
    if media is not None and worst is None and checked:
        entities = sorted({plural(o.rule.spec.entity.phrase, 2)
                           for _, result in checked for o in result.outcomes})  # fmt: skip
        sections.append(
            f"Here's what I checked: the {join_words(entities)} I judged are drawn in green."
        )
    sections += problems
    return "\n\n".join(sections) or "There was nothing to check."


# The first failure of an item, in the recording and rule with the most failures.
def _worst(checked: Sequence[tuple[Footage, Checked]]) -> tuple[Footage, Violation] | None:
    best: tuple[int, Footage, Violation] | None = None
    for footage, result in checked:
        for outcome in result.outcomes:
            items = sorted((v for v in outcome.violations if v.track_id is not None),
                           key=lambda v: v.t)  # fmt: skip
            if items and (best is None or len(items) > best[0]):
                best = (len(items), footage, items[0])
    return None if best is None else (best[1], best[2])


# No count here: tracks are not items yet (one box can be followed as several tracks), and the
# items a rule judged are counted in the next steps.
def _found_text(rules: Sequence[Rule], concepts: Sequence[str]) -> str:
    entities = sorted({rule.spec.entity.phrase for rule in rules})
    seen = set(concepts)
    found = [plural(e, 2) for e in entities if e in seen]
    if not found:
        return f"Found no {join_words([plural(e, 2) for e in entities])}"
    return f"Found the {join_words(found)}"


def _looked_text(rules: Sequence[Rule], items: int) -> str:
    parts = sorted({phrase for rule in rules for phrase in checkpoint_parts(rule.spec)})
    entities = {rule.spec.entity.phrase for rule in rules}
    what = plural(next(iter(entities)), items) if len(entities) == 1 else plural("item", items)
    return f"Looked for {join_words([plural(p, 2) for p in parts])} on {items} {what}"


def clip_buttons(footage: Footage, number: int, count: int) -> tuple[Button, ...]:
    if number >= count:
        return ()
    return (Button(label="Next clip", value=f"/clip {footage.id} {number + 1}"),)


def _missing_zones(line: Line, specs: Sequence[RuleSpec]) -> tuple[str, ...]:
    known = {zone.key for zone in line.zones} | {"view_exit"}
    needed = {key for spec in specs for key in zone_keys(spec)}
    return tuple(sorted(needed - known))


def _sample_times(span: TimeSpan, fps: float) -> list[float]:
    count = max(1, math.floor((span.end - span.start) * fps))
    return [span.start + index / fps for index in range(count)]


# A planned frame counts as cached when the store has any frame within half a sample of it.
def _cached(
    stored: Mapping[int, tuple[Observation, ...]], t: float, footage_fps: float, fps: float
) -> bool:
    low = frame_index(max(0.0, t - 0.5 / fps), footage_fps)
    high = frame_index(t + 0.5 / fps, footage_fps)
    return any(index in stored for index in range(low, high + 1))


def _spans_key(footage_id: str) -> str:
    return f"footage/{footage_id}/active_spans.json"


def _strings(value: JsonValue) -> tuple[str, ...]:
    return tuple(str(item) for item in value) if isinstance(value, list) else ()
