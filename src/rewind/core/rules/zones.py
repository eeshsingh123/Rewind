from collections.abc import Mapping
from dataclasses import dataclass

from rewind.contracts import Box, Charge, Point, Polygon, Tripwire, ViewExit, Zone
from rewind.core.ports import ConceptPrompt, Segmenter, SegmentRequest

VIEW_EXIT = Zone(key="view_exit", shape=ViewExit(), source="implicit")


class RuleError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Keyframe:
    footage_id: str
    image_key: str
    width: int
    height: int
    t: float


@dataclass(frozen=True, slots=True)
class FoundZone:
    # None when the segmenter found nothing; the call is billed either way.
    zone: Zone | None
    charges: tuple[Charge, ...]


async def find_zone(segmenter: Segmenter, key: str, phrase: str, keyframe: Keyframe) -> FoundZone:
    request = SegmentRequest(
        footage_id=keyframe.footage_id,
        media_key=keyframe.image_key,
        width=keyframe.width,
        height=keyframe.height,
        frame_times=(keyframe.t,),
        prompt=ConceptPrompt(text=phrase),
    )
    result = await segmenter.segment(request)
    if not result.observations:
        return FoundZone(zone=None, charges=result.charges)
    best = max(result.observations, key=lambda observation: observation.score)
    zone = Zone(key=key, shape=box_polygon(best.box), source="text", phrase=phrase)
    return FoundZone(zone=zone, charges=result.charges)


def box_polygon(box: Box) -> Polygon:
    corners = ((box.x0, box.y0), (box.x1, box.y0), (box.x1, box.y1), (box.x0, box.y1))
    return Polygon(points=tuple(Point(x=x, y=y) for x, y in corners))


def area(zones: Mapping[str, Zone], key: str) -> Polygon:
    shape = zones[key].shape
    if not isinstance(shape, Polygon):
        raise RuleError(f"zone {key} is a {shape.kind}, but this rule needs an area")
    return shape


def center(box: Box) -> Point:
    return Point(x=(box.x0 + box.x1) / 2, y=(box.y0 + box.y1) / 2)


def contains(polygon: Polygon, point: Point) -> bool:
    inside = False
    points = polygon.points
    for a, b in zip(points, points[1:] + points[:1], strict=True):
        if (a.y > point.y) != (b.y > point.y):
            x_at_y = a.x + (point.y - a.y) * (b.x - a.x) / (b.y - a.y)
            if point.x < x_at_y:
                inside = not inside
    return inside


# Where along start -> end (0..1) the path crosses the tripwire, or None if it does not.
def crossing(tripwire: Tripwire, start: Point, end: Point) -> float | None:
    dx, dy = end.x - start.x, end.y - start.y
    wx, wy = tripwire.end.x - tripwire.start.x, tripwire.end.y - tripwire.start.y
    denominator = dx * wy - dy * wx
    if denominator == 0:
        return None
    ox, oy = tripwire.start.x - start.x, tripwire.start.y - start.y
    along_path = (ox * wy - oy * wx) / denominator
    along_wire = (ox * dy - oy * dx) / denominator
    if 0 <= along_path <= 1 and 0 <= along_wire <= 1:
        return along_path
    return None


# The share of the part's box that lies inside the whole's box.
def overlap(part: Box, whole: Box) -> float:
    width = min(part.x1, whole.x1) - max(part.x0, whole.x0)
    height = min(part.y1, whole.y1) - max(part.y0, whole.y0)
    if width < 0 or height < 0:
        return 0.0
    part_area = (part.x1 - part.x0) * (part.y1 - part.y0)
    if part_area == 0:
        return 1.0
    return width * height / part_area
