import asyncio
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
from rewind.core.analysis.parts import (
    PartLooker,
    StillGrabber,
    checkpoint_parts,
    still_times,
    tracking_spec,
)
from rewind.core.analysis.proof import ClipMaker, ProofMarks, proof_marks
from rewind.core.analysis.replies import (
    RuleOutcome,
    clip_caption,
    duration_text,
    plural,
    rule_summary,
)
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
    ChannelError,
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
PROGRESS_EVERY_S = 30.0


class Notifier(Protocol):
    async def notify(
        self,
        channel: str,
        chat_id: str,
        text: str,
        media: tuple[ReplyMedia, ...] = (),
        buttons: tuple[Button, ...] = (),
    ) -> None: ...

    async def progress(self, channel: str, chat_id: str, key: str, text: str) -> None: ...


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


def footage_name(footage: Footage) -> str:
    return PurePosixPath(footage.original_key).name


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
        notifier: Notifier,
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
        self._notifier = notifier
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
        summary = (
            f"Got it: {duration_text(footage.duration_s)} of *{line.name}* "
            f"({footage_name(footage)}, {footage.width}x{footage.height})."
        )
        if moving >= footage.duration_s - 1:
            summary += " Something moves for the whole clip."
        elif moving == 0:
            summary += " Nothing moves in it, so there is nothing to check."
        else:
            summary += f" Something moves for {duration_text(moving)}; I'll only check that."
        rules = [rule for rule in await self._repository.rules(line.id) if rule.active]
        if rules:
            count = f"{len(rules)} {plural('rule', len(rules))}"
            next_step = f"This line has {count}. Ask me to check it."
        else:
            next_step = (
                "This line has no rules yet. Tell me what every item must have or do, "
                "one sentence per rule."
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
        return Estimate(
            tuple(estimates), missing, one_image * len(missing), on_stills,
            one_image * len(on_stills),
        )  # fmt: skip

    async def handle_analyze(self, payload: JobPayload) -> None:
        workspace_id = str(payload["workspace_id"])
        channel, chat_id = str(payload["channel"]), str(payload["chat_id"])
        job_id = str(payload["job_id"])
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
        sections: list[str] = []
        with_findings: Footage | None = None
        for index, footage_id in enumerate(footage_ids, 1):
            footage = await self._repository.get_footage(footage_id)
            if footage is None:
                continue
            name = footage_name(footage)
            of = f" ({index} of {len(footage_ids)})" if len(footage_ids) > 1 else ""
            progress = _Progress(
                self._notifier, channel, chat_id, job_id, f"Checking *{name}*{of}…"
            )
            try:
                async with progress:
                    checked = await self._check(line, footage, rules, numbers, scope)
            except BudgetExceededError as error:
                sections.append(f"I stopped before *{name}*: {error}.")
                break
            except (SegmenterError, VideoToolsError, RuleCompileError):
                logger.exception("analysis of %s failed", footage.id)
                sections.append(f"*{name}*: something went wrong while checking it. Try again.")
                continue
            line = checked.line
            spent += checked.cost
            sections.append(_footage_summary(footage, checked))
            if with_findings is None and any(o.violations for o in checked.outcomes):
                with_findings = footage
        elapsed = duration_text(time.monotonic() - started)
        text = f"Done in {elapsed} for ${spent:.2f}.\n\n" + "\n\n".join(sections)
        buttons = ()
        if with_findings is not None:
            buttons = (Button(label="Show clips", value=f"/clip {with_findings.id} 1"),)
        await self._notifier.notify(channel, chat_id, text, buttons=buttons)

    async def clip(
        self, footage: Footage, violation: Violation, number: int, count: int, max_bytes: int
    ) -> Clip:
        numbers = await self.rule_numbers(footage.line_id)
        caption = clip_caption(violation, number, count, numbers.get(violation.rule_id))
        if max_bytes <= 0:
            return Clip((), caption)
        marks_key = _marks_key(violation.id)
        marks = (
            _MARKS.validate_json(await self._blobs.read(marks_key))
            if await self._blobs.exists(marks_key)
            else ProofMarks(marks=())
        )
        # Named by what is drawn, so a rerun that changes the boxes renders a new clip.
        drawn = hashlib.sha256(_MARKS.dump_json(marks)).hexdigest()[:12]
        key = f"proof/{violation.id}-{drawn}.mp4"
        if not await self._blobs.exists(key):
            span = TimeSpan(
                start=max(0.0, violation.clip.start),
                end=min(footage.duration_s, violation.clip.end),
            )
            source = await self._blobs.local_path(footage.original_key)
            with tempfile.TemporaryDirectory() as workdir:
                destination = Path(workdir) / "clip.mp4"
                if marks.marks:
                    await self._clips.make_clip(source, destination, span, marks.marks)
                else:
                    await self._video.cut_clip(source, destination, span)
                await self._blobs.write_file(key, destination)
        size = (await self._blobs.local_path(key)).stat().st_size
        if size > max_bytes:
            return Clip((), caption + "\n(The clip is too big to send on this chat.)")
        return Clip((ReplyMedia(kind="video", blob_key=key, caption=caption),), caption)

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
    ) -> Checked:
        spans = await self.active_spans(footage)
        tracked = tuple(tracking_spec(rule.spec) for rule in rules)
        plan = self._planner.plan(
            self._settings.strategy, PlanInput(footage, spans, tracked, self._settings.fps)
        )
        result = await self._runner.run(footage, plan, self._segmenters.video(scope))
        charges = list(result.charges)
        line, zone_charges = await self._find_zones(line, footage, spans, rules, scope)
        charges += zone_charges
        tracks = naive_tracks(footage.id, result.observations, max(1.0, 3 / self._settings.fps))
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
        looks = await self._parts.look(footage, times, self._segmenters.image(scope))
        charges += looks.charges
        scene = dataclasses.replace(scene, observations=scene.observations + looks.observations)
        outcomes: list[RuleOutcome] = []
        for rule in rules:
            if rule.id not in judged:
                continue
            found = evaluate(rule.id, rule.spec, scene)
            await self._save_marks(rule.spec, scene, judged[rule.id], found)
            number = numbers.get(rule.id, 0)
            outcomes.append(RuleOutcome(number, rule, len(judged[rule.id]), found))
        violations = [v for outcome in outcomes for v in outcome.violations]
        await self._repository.replace_violations(footage.id, list(judged), violations)
        cost = CostEstimate(charges=tuple(charges)).total_usd
        return Checked(line, tuple(outcomes), cost, tuple(skipped))

    # What each proof clip draws is saved now, while the tracks are at hand, so a clip asked
    # for later shows exactly the boxes this check decided on.
    async def _save_marks(
        self,
        spec: RuleSpec,
        scene: Scene,
        judged: Sequence[Check],
        found: Sequence[Violation],
    ) -> None:
        items = {
            item.track.track_id: item
            for item in build_items(spec, scene.tracks, scene.observations, {})
        }
        failed = {v.track_id: v for v in found if v.track_id is not None}
        for violation in found:
            if violation.track_id is None:
                continue
            marks = proof_marks(violation, items, judged, failed)
            await self._blobs.write(_marks_key(violation.id), _MARKS.dump_json(marks))

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


# A check can take minutes, so its progress message says how long it has run, every half
# minute. Only a cue: if the chat refuses an update, the check goes on.
class _Progress:
    def __init__(self, notifier: Notifier, channel: str, chat_id: str, key: str, text: str) -> None:
        self._notifier = notifier
        self._channel = channel
        self._chat_id = chat_id
        self._key = key
        self._text = text
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> None:
        await self._show(self._text)
        self._task = asyncio.create_task(self._tick())

    async def __aexit__(self, *exc_info: object) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.wait([self._task])

    async def _tick(self) -> None:
        started = time.monotonic()
        while True:
            await asyncio.sleep(PROGRESS_EVERY_S)
            elapsed = duration_text(time.monotonic() - started)
            try:
                await self._show(f"{self._text} {elapsed} so far.")
            except ChannelError:
                logger.warning("could not update progress in chat %s", self._chat_id, exc_info=True)
                return

    async def _show(self, text: str) -> None:
        await self._notifier.progress(self._channel, self._chat_id, self._key, text)


def _footage_summary(footage: Footage, checked: Checked) -> str:
    parts = [f"*{footage_name(footage)}* ({duration_text(footage.duration_s)})"]
    parts += [rule_summary(outcome) for outcome in checked.outcomes]
    parts += checked.skipped
    return "\n\n".join(parts)


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


def _marks_key(violation_id: str) -> str:
    return f"proof/{violation_id}.json"


def _strings(value: JsonValue) -> tuple[str, ...]:
    return tuple(str(item) for item in value) if isinstance(value, list) else ()
