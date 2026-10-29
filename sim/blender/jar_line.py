import random

from common import (
    Scene,
    box,
    cylinder,
    empty,
    image_material,
    key_pop,
    key_x,
    radians,
    wrap_uv,
)

CONTENTS = {
    "strawberry jam": (0.45, 0.03, 0.05),
    "orange marmalade": (0.85, 0.38, 0.04),
    "honey": (0.80, 0.52, 0.08),
    "pickles": (0.35, 0.45, 0.12),
}
CAP_COLOURS = {
    "gold": (0.80, 0.62, 0.25),
    "red": (0.60, 0.06, 0.05),
    "white": (0.85, 0.85, 0.82),
    "black": (0.03, 0.03, 0.03),
    "green": (0.10, 0.35, 0.12),
}


def build(scene: Scene) -> None:
    spec = scene.spec
    top = spec["belt"]["top"]
    _guides(scene, top)
    for station in spec["stations"]:
        if station["key"] == "capper":
            _capper(scene, station, top)
        else:
            _labeller(scene, station, top)
    for item in spec["items"]:
        _jar(scene, item, top)


def _jar(scene: Scene, item: dict, top: float) -> None:
    width, _, height = scene.spec["item_size"]
    radius = width / 2
    style = item["style"]
    rng = random.Random(item["id"])
    root = empty(item["id"], (item["offset"], rng.uniform(-0.01, 0.01), top))
    root.rotation_euler = (0, 0, radians(rng.uniform(-25, 25)))
    key_x(scene, root, item["offset"])
    body_height = height * 0.78
    contents = scene.flat(
        f"contents_{style['contents']}", CONTENTS[style["contents"]], roughness=0.25, coat=1.0
    )
    body = cylinder(
        f"{item['id']}_body",
        radius * 0.94,
        body_height * 0.92,
        (0, 0, body_height * 0.47),
        contents,
        root,
    )
    glass = scene.flat("glass", (0.6, 0.7, 0.68), roughness=0.02, coat=1.0, alpha=0.1)
    shell = cylinder(
        f"{item['id']}_glass", radius, body_height, (0, 0, body_height / 2), glass, root
    )
    for target in (body, shell):
        bevel = target.modifiers.new("bevel", "BEVEL")
        bevel.width = 0.008
        bevel.segments = 3
    neck_height = height * 0.12
    cylinder(
        f"{item['id']}_neck",
        radius * 0.82,
        neck_height,
        (0, 0, body_height + neck_height / 2),
        glass,
        root,
    )
    label = item["parts"].get("label")
    sleeve = cylinder(
        f"{item['id']}_label",
        radius * 1.02,
        body_height * 0.5,
        (0, 0, body_height * 0.45),
        image_material(f"{item['id']}_label", item["label_image"]),
        root,
        caps=False,
    )
    wrap_uv(sleeve)
    key_pop(scene, sleeve, label)
    cap_colour = scene.flat(
        f"cap_{style['cap_colour']}", CAP_COLOURS[style["cap_colour"]], roughness=0.3, metallic=0.85
    )
    cap = cylinder(
        f"{item['id']}_cap",
        radius * 0.9,
        height * 0.12,
        (0, 0, body_height + neck_height + height * 0.04),
        cap_colour,
        root,
    )
    key_pop(scene, cap, item["parts"].get("cap"), ramp_s=0.15)


def _guides(scene: Scene, top: float) -> None:
    steel = scene.textured("steel", 1.0, metallic=1.0)
    width = scene.spec["belt"]["width"]
    for side in (-1, 1):
        guide = cylinder(
            f"guide_{side}", 0.006, 7.0, (0, side * (width / 2 - 0.03), top + 0.07), steel
        )
        guide.rotation_euler = (0, radians(90), 0)


def _capper(scene: Scene, station: dict, top: float) -> None:
    painted = scene.textured("painted", 1.0, tuple(scene.spec["machine_colour"]), grey=True)
    steel = scene.textured("steel", 1.0, metallic=1.0)
    width = scene.spec["belt"]["width"]
    x = (station["start"] + station["end"]) / 2
    # Cantilevered from a column behind the belt, so nothing stands between camera and jars.
    box("capper_column", (0.1, 0.1, 0.75), (x, width / 2 + 0.12, top + 0.25), painted, bevel=0.01)
    box(
        "capper_arm",
        (0.12, width / 2 + 0.16, 0.08),
        (x, width / 4 + 0.06, top + 0.46),
        painted,
        bevel=0.01,
    )
    box(
        "capper_hopper",
        (0.18, 0.14, 0.16),
        (x - 0.02, width / 2 + 0.12, top + 0.68),
        steel,
        bevel=0.01,
    )
    head = cylinder("capper_head", 0.05, 0.12, (x, 0, top + 0.3), steel)
    rest, down = top + 0.3, top + 0.215
    for t in station["times"]:
        if t < -1 or t > scene.spec["render"]["frames"] / scene.fps + 1:
            continue
        for at, z in ((t - 0.35, rest), (t - 0.05, down), (t + 0.25, rest)):
            head.location.z = z
            head.keyframe_insert("location", index=2, frame=scene.frame(at))


def _labeller(scene: Scene, station: dict, top: float) -> None:
    painted = scene.textured("painted", 1.0, tuple(scene.spec["machine_colour"]), grey=True)
    width = scene.spec["belt"]["width"]
    x = (station["start"] + station["end"]) / 2
    y = width / 2 + 0.18
    box("labeller_body", (0.34, 0.22, 0.3), (x, y, top + 0.05), painted, bevel=0.01)
    paper = scene.flat("label_roll", (0.92, 0.92, 0.88), roughness=0.6)
    for offset in (-0.08, 0.08):
        cylinder(f"labeller_roll_{offset}", 0.07, 0.05, (x + offset, y, top + 0.225), paper)
