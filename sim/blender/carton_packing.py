import random

import bpy
from common import (
    Scene,
    box,
    cylinder,
    empty,
    image_material,
    key_pop,
    key_x,
    radians,
)

TAPE_COLOURS = {
    "brown": (0.55, 0.36, 0.16),
    "clear": (0.75, 0.68, 0.52),
    "white": (0.85, 0.85, 0.82),
}
OPEN_DEGREES = 115
# Minor (end) flaps fold first, then the major flaps close over them, finishing at the part time.
MINOR_FOLD = (-1.4, -0.7)
MAJOR_FOLD = (-0.7, 0.0)
LABEL_SIZE = (0.16, 0.12)
TAPE_WIDTH = 0.05


def build(scene: Scene) -> None:
    top = scene.spec["belt"]["top"]
    for station in scene.spec["stations"]:
        {"folder": _folder, "labeller": _labeller, "sealing": _taper}[station["key"]](
            scene, station, top
        )
    for item in scene.spec["items"]:
        _carton(scene, item, top)


def _carton(scene: Scene, item: dict, top: float) -> None:
    length, width, height = scene.spec["item_size"]
    style = item["style"]
    rng = random.Random(item["id"])
    root = empty(item["id"], (item["offset"], rng.uniform(-0.015, 0.015), top))
    root.rotation_euler = (0, 0, radians(rng.uniform(-3, 3)))
    key_x(scene, root, item["offset"])
    tint = style["cardboard_tint"]
    cardboard = scene.textured("cardboard", 1.5, (tint, tint * 0.97, tint * 0.92))
    body = _open_box(f"{item['id']}_body", length, width, height, cardboard)
    body.parent = root
    flaps = _flaps(item, length, width, height, cardboard, root)
    folded_at = item["parts"].get("flaps")
    for name, (flap, axis, sign, minor) in flaps.items():
        stays_open = folded_at is None and name == style["open_flap"]
        _key_flap(scene, flap, axis, sign, minor, item["station_times"]["folder"], stays_open)
    _tape(scene, item, root, length, height)
    _label(scene, item, root, width, height)


def _open_box(name, length, width, height, material):
    mesh = bpy.data.meshes.new(name)
    x, y = length / 2, width / 2
    vertices = [(-x, -y, 0), (x, -y, 0), (x, y, 0), (-x, y, 0)]
    vertices += [(-x, -y, height), (x, -y, height), (x, y, height), (-x, y, height)]
    faces = [(0, 3, 2, 1), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh.from_pydata(vertices, [], faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(material)
    solidify = obj.modifiers.new("walls", "SOLIDIFY")
    solidify.thickness = 0.005
    bevel = obj.modifiers.new("bevel", "BEVEL")
    bevel.width = 0.003
    return obj


def _flaps(item, length, width, height, material, root):
    depth = width / 2
    # name -> (object, rotation axis, open sign, is a minor flap)
    specs = {
        "front": ((length, depth), (0, -width / 2, height + 0.006), (0, depth / 2), 0, 1, False),
        "back": ((length, depth), (0, width / 2, height + 0.006), (0, -depth / 2), 0, -1, False),
        "left": ((depth, width), (-length / 2, 0, height + 0.002), (depth / 2, 0), 1, -1, True),
        "right": ((depth, width), (length / 2, 0, height + 0.002), (-depth / 2, 0), 1, 1, True),
    }
    flaps = {}
    for name, (size, hinge, centre, axis, sign, minor) in specs.items():
        pivot = empty(f"{item['id']}_{name}_hinge", hinge)
        pivot.parent = root
        box(
            f"{item['id']}_{name}_flap",
            (size[0], size[1], 0.004),
            (centre[0], centre[1], 0),
            material,
            pivot,
        )
        flaps[name] = (pivot, axis, sign, minor)
    return flaps


def _key_flap(scene, pivot, axis, sign, minor, folder_t, stays_open):
    opened = radians(OPEN_DEGREES) * sign
    pivot.rotation_euler[axis] = opened
    if stays_open:
        return
    start, end = MINOR_FOLD if minor else MAJOR_FOLD
    pivot.keyframe_insert("rotation_euler", index=axis, frame=scene.frame(folder_t + start))
    pivot.rotation_euler[axis] = 0.0
    pivot.keyframe_insert("rotation_euler", index=axis, frame=scene.frame(folder_t + end))


def _tape(scene, item, root, length, height):
    applied = item["parts"].get("tape")
    if applied is None:
        return
    tape_material = scene.flat(
        f"tape_{item['style']['tape_colour']}",
        TAPE_COLOURS[item["style"]["tape_colour"]],
        roughness=0.2,
        coat=0.6,
    )
    # Anchored at the leading (+x) end, so it grows backwards as the carton passes the roller.
    anchor = empty(f"{item['id']}_tape_anchor", (length / 2 + 0.002, 0, height + 0.012))
    anchor.parent = root
    box(f"{item['id']}_tape", (length + 0.004, TAPE_WIDTH, 0.0015), (-length / 2, 0, 0), tape_material, anchor)
    pass_s = length / scene.spec["belt"]["speed"]
    if applied - pass_s / 2 <= 0:
        return
    anchor.scale = (0.0, 1.0, 1.0)
    anchor.keyframe_insert("scale", index=0, frame=scene.frame(applied - pass_s / 2))
    anchor.scale = (1.0, 1.0, 1.0)
    anchor.keyframe_insert("scale", index=0, frame=scene.frame(applied + pass_s / 2))


def _label(scene, item, root, width, height):
    applied = item["parts"].get("label")
    if applied is None:
        return
    image = item["label_image"] if item["parts"].get("qr") is not None else item["label_plain"]
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0.02, -width / 2 - 0.004, height * 0.5))
    label = bpy.context.active_object
    label.name = f"{item['id']}_label"
    label.scale = (LABEL_SIZE[0], LABEL_SIZE[1], 1)
    label.rotation_euler = (radians(90), 0, 0)
    label.data.materials.append(image_material(label.name, image, roughness=0.5))
    label.parent = root
    key_pop(scene, label, applied, ramp_s=0.25)


def _folder(scene, station, top):
    steel = scene.textured("steel", 1.0, metallic=1.0)
    painted = scene.textured("painted", 1.0, tuple(scene.spec["machine_colour"]), grey=True)
    width = scene.spec["belt"]["width"]
    length = station["end"] - station["start"]
    x = (station["start"] + station["end"]) / 2
    height = scene.spec["item_size"][2]
    # Machines hang from columns behind the belt, so nothing stands between camera and cartons.
    box("folder_column", (0.08, 0.08, height + 0.75), (x, width / 2 + 0.12, (top + height + 0.4) / 2 + 0.2), painted)
    for side in (-1, 1):
        rail = cylinder(f"folder_rail_{side}", 0.008, length + 0.3, (x, side * 0.05, top + height + 0.05), steel)
        rail.rotation_euler = (0, radians(90), radians(side * 6))
    box("folder_beam", (0.08, width / 2 + 0.16, 0.06), (x, width / 4 + 0.06, top + height + 0.33), painted)


def _labeller(scene, station, top):
    painted = scene.textured("painted", 1.0, tuple(scene.spec["machine_colour"]), grey=True)
    steel = scene.textured("steel", 1.0, metallic=1.0)
    width = scene.spec["belt"]["width"]
    x = (station["start"] + station["end"]) / 2
    height = scene.spec["item_size"][2]
    # Mounted low on the camera side so it hides the belt frame, not the cartons.
    box("labeller_body", (0.36, 0.2, 0.3), (x, -width / 2 - 0.22, top - 0.2), painted, bevel=0.01)
    pad = box("labeller_pad", (0.12, 0.04, 0.09), (x, -width / 2 - 0.12, top + height * 0.5), steel)
    rest, pressed = -width / 2 - 0.12, -width / 2 - 0.035
    for t in station["times"]:
        for at, y in ((t - 0.45, rest), (t - 0.05, pressed), (t + 0.2, rest)):
            pad.location.y = y
            pad.keyframe_insert("location", index=1, frame=scene.frame(at))
    box("labeller_stand", (0.08, 0.08, top - 0.35), (x, -width / 2 - 0.22, (top - 0.35) / 2), steel)
    box("labeller_arm", (0.04, 0.04, height * 0.5 + 0.12), (x, -width / 2 - 0.16, top - 0.02), steel)


def _taper(scene, station, top):
    painted = scene.textured("painted", 1.0, tuple(scene.spec["machine_colour"]), grey=True)
    width = scene.spec["belt"]["width"]
    x = (station["start"] + station["end"]) / 2
    height = scene.spec["item_size"][2]
    span = station["end"] - station["start"]
    box("taper_column", (0.1, 0.1, height + 0.8), (x, width / 2 + 0.14, (top + height + 0.4) / 2 + 0.2), painted)
    box("taper_head", (span, 0.14, 0.12), (x, 0, top + height + 0.16), painted, bevel=0.01)
    roll = cylinder("tape_roll", 0.08, TAPE_WIDTH + 0.01, (x + 0.05, 0, top + height + 0.3), scene.flat("tape_roll", (0.55, 0.36, 0.16), roughness=0.3))
    roll.rotation_euler = (radians(90), 0, 0)
    core = cylinder("tape_core", 0.035, TAPE_WIDTH + 0.012, (x + 0.05, 0, top + height + 0.3), scene.flat("tape_core", (0.7, 0.6, 0.45)))
    core.rotation_euler = (radians(90), 0, 0)
