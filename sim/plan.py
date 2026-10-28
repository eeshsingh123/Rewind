import math
import random
from dataclasses import dataclass

from rewind.contracts import (
    Camera,
    GroundTruth,
    GroundTruthItem,
    ScenarioConfig,
    ScenarioEvent,
    TimeSpan,
    Zone,
)
from rewind.contracts.scenario import PropValue
from sim.kinematics import Belt, merge_spans
from sim.layout import EXIT_MARGIN_S, ItemTimeline, Line
from sim.lines import LINES
from sim.projection import ground_polygon, view_edge_x

VIEW_EXIT = "view_exit"
MIN_GAP_M = 0.04


@dataclass(frozen=True, slots=True)
class Look:
    hdri_strength: float
    hdri_rotation: float
    light_energy: float
    # -1 is cool white, +1 is warm white.
    light_warmth: float
    shake_px: float
    noise: float
    blur: float
    crf: int
    props_seed: int


@dataclass(frozen=True, slots=True)
class ItemPlan:
    id: str
    order_id: str | None
    # Where the item's centre is along the belt at t = 0; it moves with the belt from there.
    offset: float
    timeline: ItemTimeline
    # When the item's centre reaches each station, even before the footage starts.
    station_times: dict[str, float]
    style: dict[str, PropValue]
    violations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScenePlan:
    config: ScenarioConfig
    line: Line
    camera: Camera
    belt: Belt
    items: tuple[ItemPlan, ...]
    zones: tuple[Zone, ...]
    look: Look


def plan_scene(config: ScenarioConfig) -> ScenePlan:
    line = LINES[config.builder]
    unknown = sorted(set(config.faults) - line.faults)
    if unknown:
        raise ValueError(f"{config.builder} has no faults named {', '.join(unknown)}")
    aspect = config.width / config.height
    camera = _camera(config)
    speed_rng = _rng(config, "speed")
    speed = _param(config, "speed_m_s", 0.1) * speed_rng.uniform(
        1 - _param(config, "speed_jitter", 0.1), 1 + _param(config, "speed_jitter", 0.1)
    )
    belt = Belt(speed=speed, jams=_jams(config))
    enter_x = view_edge_x(camera, 0.0, line.item_center_z, aspect, 0.0)
    exit_x = view_edge_x(camera, 0.0, line.item_center_z, aspect, 1.0)
    for station in line.stations:
        if not enter_x < station.start < station.end < exit_x:
            raise ValueError(f"station {station.key} is outside the view of this camera")
    items = _items(config, line, belt, enter_x, exit_x)
    zones = tuple(
        Zone(
            key=station.key,
            shape=ground_polygon(
                camera,
                station.start,
                station.end,
                -line.belt_width / 2,
                line.belt_width / 2,
                line.item_center_z,
                aspect,
            ),
            source="drawn",
        )
        for station in line.stations
    )
    return ScenePlan(config, line, camera, belt, items, zones, _look(config))


def ground_truth(plan: ScenePlan) -> GroundTruth:
    config = plan.config
    items = tuple(
        GroundTruthItem(
            id=item.id,
            order_id=item.order_id,
            t_spawn=round(item.timeline.first_seen, 3),
            t_cross={key: round(t, 3) for key, t in _crossings(item, config.duration_s).items()},
            props=_props(item),
            expected_violations=item.violations,
        )
        for item in plan.items
    )
    events = tuple(
        ScenarioEvent(
            kind="jam", span=TimeSpan(start=jam.start, end=min(jam.end, config.duration_s))
        )
        for jam in plan.belt.jams
        if jam.start < config.duration_s
    )
    return GroundTruth(
        scenario=config.name,
        seed=config.seed,
        duration_s=config.duration_s,
        fps=config.fps,
        width=config.width,
        height=config.height,
        camera=plan.camera,
        zones=plan.zones,
        items=items,
        events=events,
    )


def _items(
    config: ScenarioConfig, line: Line, belt: Belt, enter_x: float, exit_x: float
) -> tuple[ItemPlan, ...]:
    spacing_rng, fault_rng, style_rng = (_rng(config, s) for s in ("spacing", "faults", "style"))
    jitter = _param(config, "interval_jitter", 0.15)
    travel = belt.distance(config.duration_s)
    first_order = config.params.get("first_order_id")
    planned: list[ItemPlan] = []
    offset = exit_x
    while offset + travel > enter_x - line.item_size[0]:
        gap = belt.speed * config.item_interval_s * spacing_rng.uniform(1 - jitter, 1 + jitter)
        offset -= max(gap, line.item_size[0] + MIN_GAP_M)
        faults = frozenset(f for f, p in sorted(config.faults.items()) if fault_rng.random() < p)
        style = line.style(style_rng)
        if offset + travel <= enter_x:
            continue
        number = len(planned) + 1
        order_id = None if first_order is None else str(int(first_order) + number - 1)
        timeline = _timeline(config, line, belt, offset, enter_x, exit_x, faults)
        planned.append(
            ItemPlan(
                id=f"{line.item_kind}-{number:03d}",
                order_id=order_id,
                offset=offset,
                timeline=timeline,
                station_times={s.key: belt.time_at(s.x - offset) for s in line.stations},
                style=style,
                violations=line.judge(timeline),
            )
        )
    return tuple(planned)


def _timeline(
    config: ScenarioConfig,
    line: Line,
    belt: Belt,
    offset: float,
    enter_x: float,
    exit_x: float,
    faults: frozenset[str],
) -> ItemTimeline:
    duration = config.duration_s
    exit_t = belt.time_at(exit_x - offset)
    entry: dict[str, float] = {}
    stay: dict[str, float] = {}
    parts: dict[str, float | None] = {}
    for station in line.stations:
        start, end = belt.time_at(station.start - offset), belt.time_at(station.end - offset)
        if end > 0 and start < duration:
            entry[station.key] = max(0.0, start)
            stay[station.key] = min(end, duration) - entry[station.key]
        for part in station.parts:
            skipped = part.fault in faults
            parts[part.key] = None if skipped else belt.time_at(station.x - offset)
    return ItemTimeline(
        first_seen=max(0.0, belt.time_at(enter_x - offset)),
        last_seen=min(exit_t, duration),
        exits=exit_t < duration - EXIT_MARGIN_S,
        zone_entry=entry,
        zone_stay=stay,
        parts=parts,
        faults=faults,
    )


def _crossings(item: ItemPlan, duration_s: float) -> dict[str, float]:
    crossings = dict(item.timeline.zone_entry)
    if item.timeline.last_seen < duration_s:
        crossings[VIEW_EXIT] = item.timeline.last_seen
    return crossings


def _props(item: ItemPlan) -> dict[str, PropValue]:
    timeline = item.timeline
    props: dict[str, PropValue] = dict(item.style)
    props.update({part: timeline.has(part, timeline.last_seen) for part in timeline.parts})
    props.update({f"{zone}_dwell_s": round(s, 3) for zone, s in timeline.zone_stay.items()})
    return props


def _camera(config: ScenarioConfig) -> Camera:
    rng = _rng(config, "camera")
    move = _param(config, "camera_jitter_m", 0.05)
    aim = move / 2
    base = config.camera
    return Camera(
        position=_jitter(base.position, move, rng),
        look_at=_jitter(base.look_at, aim, rng),
        focal_mm=base.focal_mm + rng.uniform(-1, 1) * _param(config, "focal_jitter_mm", 1.0),
    )


def _jams(config: ScenarioConfig) -> tuple[TimeSpan, ...]:
    rng = _rng(config, "jams")
    spans = list(config.jams)
    shortest, longest = _param(config, "jam_min_s", 6.0), _param(config, "jam_max_s", 14.0)
    if rng.random() < _param(config, "jam_probability", 0.0):
        length = rng.uniform(shortest, longest)
        start = rng.uniform(5.0, max(5.0, config.duration_s - length - 5.0))
        spans.append(TimeSpan(start=round(start, 2), end=round(start + length, 2)))
    return merge_spans(spans)


def _look(config: ScenarioConfig) -> Look:
    rng = _rng(config, "look")
    return Look(
        hdri_strength=rng.uniform(0.35, 0.9),
        hdri_rotation=rng.uniform(0, 2 * math.pi),
        light_energy=rng.uniform(0.75, 1.3),
        light_warmth=rng.uniform(-1, 1),
        shake_px=rng.uniform(1.0, 3.0),
        noise=rng.uniform(5, 12),
        blur=rng.uniform(0.4, 0.9),
        crf=rng.randint(27, 31),
        props_seed=rng.randrange(1_000_000),
    )


def _jitter(
    vector: tuple[float, float, float], size: float, rng: random.Random
) -> tuple[float, float, float]:
    x, y, z = (round(v + rng.uniform(-size, size), 4) for v in vector)
    return x, y, z


def _rng(config: ScenarioConfig, purpose: str) -> random.Random:
    # One stream per purpose, so tuning one parameter does not reshuffle the others.
    return random.Random(f"{config.seed}:{purpose}")


def _param(config: ScenarioConfig, key: str, default: float) -> float:
    value = config.params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"param {key} must be a number")
    return float(value)
