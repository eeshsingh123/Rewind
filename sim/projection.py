import math

from rewind.contracts import Camera, Point, Polygon

Vector3 = tuple[float, float, float]

# Blender's default sensor, fitted to the width of a landscape frame.
SENSOR_WIDTH_MM = 36.0
WORLD_UP: Vector3 = (0.0, 0.0, 1.0)


def project(camera: Camera, point: Vector3, aspect: float) -> tuple[float, float] | None:
    # Returns (u, v) with the origin at the top left of the frame, or None behind the camera.
    forward, right, up = _basis(camera)
    offset = _sub(point, camera.position)
    depth = _dot(offset, forward)
    if depth <= 0:
        return None
    scale = camera.focal_mm / SENSOR_WIDTH_MM
    u = 0.5 + scale * _dot(offset, right) / depth
    v = 0.5 - scale * aspect * _dot(offset, up) / depth
    return u, v


def view_edge_x(camera: Camera, y: float, z: float, aspect: float, edge: float) -> float:
    # The x along the line (x, y, z) that projects onto the frame column u = edge.
    low, high = -50.0, 50.0
    for _ in range(80):
        middle = (low + high) / 2
        projected = project(camera, (middle, y, z), aspect)
        if projected is not None and projected[0] < edge:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def ground_polygon(
    camera: Camera, x0: float, x1: float, y0: float, y1: float, z: float, aspect: float
) -> Polygon:
    corners: list[Point] = []
    for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        projected = project(camera, (x, y, z), aspect)
        if projected is None:
            raise ValueError("zone corner is behind the camera")
        u, v = projected
        corners.append(Point(x=min(1.0, max(0.0, u)), y=min(1.0, max(0.0, v))))
    return Polygon(points=tuple(corners))


def _basis(camera: Camera) -> tuple[Vector3, Vector3, Vector3]:
    forward = _normalize(_sub(camera.look_at, camera.position))
    right = _normalize(_cross(forward, WORLD_UP))
    return forward, right, _cross(right, forward)


def _sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _normalize(a: Vector3) -> Vector3:
    length = math.sqrt(_dot(a, a))
    return (a[0] / length, a[1] / length, a[2] / length)
