import logging
import math
import tempfile
import time
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
from rewind.core.analysis.tracks import item_label, naive_tracks
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
    Keyframe,
    RuleCompileError,
    RuleCompiler,
    RuleError,
    Scene,
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


class StillGrabber(Protocol):
    async def still(self, source: Path, t: float, destination: Path) -> None:
        """Writes the frame at footage time t as a JPEG."""
        ...


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

    @property
    def total_usd(self) -> Decimal:
        return sum((f.usd for f in self.footage), self.zones_usd)


@dataclass(frozen=True, slots=True)
class Clip:
    # Empty when the chat cannot take the clip.
    media: tuple[ReplyMedia, ...]
    caption: str


def footage_name(footage: Footage) -> str:
    return PurePosixPath(footage.original_key).name


def clock_time(t: float) -> str:
    minutes, seconds = divmod(int(t), 60)
    return f"{minutes:02d}:{seconds:02d}"


def duration_text(seconds: float) -> str:
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes} min {rest} s" if minutes else f"{rest} s"


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
        self._notifier = notifier
        self._settings = settings

    async def handle_ingest(self, payload: JobPayload) -> None:
        channel, chat_id = str(payload["channel"]), str(payload["chat_id"])
        line = await self._repository.get_line(str(payload["line_id"]))
        if line is None:
            return
        drawn = tuple(zone for zone in line.zones if zone.source == "drawn")
        try:
            ingested = await self._ingest.ingest(
                str(payload["footage_id"]), line.id, str(payload["original_key"]), drawn
            )
        except VideoToolsError:
            logger.exception("ingest failed")
            await self._notifier.notify(
                channel, chat_id, "I couldn't read that video. Is it a normal MP4 or MOV file?"
            )
            return
        footage = ingested.footage
        await self._blobs.write(_spans_key(footage.id), _SPANS.dump_json(ingested.active_spans))
        await self._repository.save_footage(footage)
        moving = sum(span.end - span.start for span in ingested.active_spans)
        parts = [
            f"Got it: {duration_text(footage.duration_s)} of *{line.name}* "
            f"({footage_name(footage)}, {footage.width}x{footage.height})."
        ]
        if moving >= footage.duration_s - 1:
            parts.append("Something moves for the whole clip.")
        elif moving == 0:
            parts.append("Nothing moves in it, so there is nothing to check.")
        else:
            parts.append(f"Something moves for {duration_text(moving)}; I'll only check that.")
        rules = [rule for rule in await self._repository.rules(line.id) if rule.active]
        if rules:
            parts.append(f"\n\nThis line has {len(rules)} rules. Ask me to check it.")
        else:
            parts.append(
                "\n\nThis line has no rules yet. Tell me what every item must have or do, "
                "one sentence per rule."
            )
        await self._notifier.notify(channel, chat_id, " ".join(parts).replace(" \n", "\n"))

    async def estimate(
        self, workspace_id: str, line: Line, footage: Sequence[Footage], rules: Sequence[Rule]
    ) -> Estimate:
        specs = tuple(rule.spec for rule in rules)
        segmenter = self._segmenters.video(BudgetScope(workspace_id))
        estimates = [await self._estimate_footage(footage_item, specs, segmenter)
                     for footage_item in footage]  # fmt: skip
        missing = _missing_zones(line, specs)
        zones_usd = Decimal(0)
        if missing and footage:
            image = self._segmenters.image(BudgetScope(workspace_id))
            request = SegmentRequest(footage[0].id, "still.jpg", 1, 1, (0.0,), _NO_PROMPT)
            zones_usd = CostEstimate(charges=image.estimate(request)).total_usd * len(missing)
        return Estimate(tuple(estimates), missing, zones_usd)

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
        summaries: list[str] = []
        first: tuple[Footage, Violation, int] | None = None
        for index, footage_id in enumerate(footage_ids, 1):
            footage = await self._repository.get_footage(footage_id)
            if footage is None:
                continue
            name = footage_name(footage)
            await self._notifier.progress(
                channel, chat_id, job_id,
                f"Checking *{name}* ({index} of {len(footage_ids)})…",
            )  # fmt: skip
            try:
                line, violations, items, cost, skipped = await self._check(
                    line, footage, rules, scope
                )
            except BudgetExceededError as error:
                summaries.append(f"I stopped before *{name}*: {error}.")
                break
            except (SegmenterError, VideoToolsError, RuleCompileError) as error:
                logger.exception("analysis of %s failed", footage.id)
                summaries.append(f"*{name}*: the check failed ({type(error).__name__}).")
                continue
            spent += cost
            summaries.append(_summary(name, violations, items, numbers, skipped))
            if first is None and violations:
                first = (footage, violations[0], len(violations))
        elapsed = duration_text(time.monotonic() - started)
        text = f"Done in {elapsed} for ${spent:.2f}.\n\n" + "\n\n".join(summaries)
        await self._notifier.notify(channel, chat_id, text)
        if first is not None:
            footage, violation, count = first
            clip = await self.clip(footage, violation, 1, count, _max_bytes(payload))
            await self._notifier.notify(
                channel, chat_id, clip.caption, clip.media, clip_buttons(footage, 1, count)
            )

    async def clip(
        self, footage: Footage, violation: Violation, number: int, count: int, max_bytes: int
    ) -> Clip:
        caption = (
            f"{number} of {count} · {clock_time(violation.t)} · "
            f"{item_label(violation.track_id)}: {violation.reason}"
        )
        if max_bytes <= 0:
            return Clip((), caption)
        key = f"clips/{violation.id}.mp4"
        if not await self._blobs.exists(key):
            span = TimeSpan(
                start=max(0.0, violation.clip.start),
                end=min(footage.duration_s, violation.clip.end),
            )
            source = await self._blobs.local_path(footage.original_key)
            with tempfile.TemporaryDirectory() as workdir:
                destination = Path(workdir) / "clip.mp4"
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
        self, line: Line, footage: Footage, rules: Sequence[Rule], scope: BudgetScope
    ) -> tuple[Line, tuple[Violation, ...], int, Decimal, list[str]]:
        specs = tuple(rule.spec for rule in rules)
        spans = await self.active_spans(footage)
        plan = self._planner.plan(
            self._settings.strategy, PlanInput(footage, spans, specs, self._settings.fps)
        )
        result = await self._runner.run(footage, plan, self._segmenters.video(scope))
        cost = CostEstimate(charges=result.charges).total_usd
        line, zone_charges = await self._find_zones(line, footage, spans, rules, scope)
        cost += CostEstimate(charges=zone_charges).total_usd
        tracks = naive_tracks(footage.id, result.observations, max(1.0, 3 / self._settings.fps))
        scene = Scene(footage.id, footage.duration_s, tracks, result.observations, line.zones)
        violations: list[Violation] = []
        skipped: list[str] = []
        for rule in rules:
            try:
                violations += evaluate(rule.id, rule.spec, scene)
            except RuleError as error:
                skipped.append(f"{rule.text} ({error})")
        await self._repository.save_violations(violations)
        entities = {spec.entity.phrase for spec in specs}
        items = sum(1 for track in tracks if track.concept in entities)
        ordered = tuple(sorted(violations, key=lambda v: (v.t, v.id)))
        return line, ordered, items, cost, skipped

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
        plan = self._planner.plan(
            self._settings.strategy, PlanInput(footage, spans, specs, self._settings.fps)
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


def _summary(
    name: str,
    violations: Sequence[Violation],
    items: int,
    numbers: dict[str, int],
    skipped: Sequence[str],
) -> str:
    if not violations:
        text = f"*{name}*: no violations in {items} items."
    else:
        listed = "\n".join(
            f"{index}. {violation_line(v, numbers)}" for index, v in enumerate(violations, 1)
        )
        text = f"*{name}*: *{len(violations)} violations in {items} items:*\n{listed}"
    if skipped:
        text += "\nNot checked: " + "; ".join(skipped)
    return text


def violation_line(violation: Violation, numbers: dict[str, int]) -> str:
    number = numbers.get(violation.rule_id)
    rule = f"rule {number}" if number else "a removed rule"
    return (
        f"{clock_time(violation.t)} · {item_label(violation.track_id)}: {violation.reason} ({rule})"
    )


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


def _max_bytes(payload: JobPayload) -> int:
    value = payload.get("max_media_bytes", 0)
    return value if isinstance(value, int) else 0
