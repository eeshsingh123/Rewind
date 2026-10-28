import json
import random
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).parent))

import carton_packing  # noqa: E402
import jar_line  # noqa: E402
from common import (  # noqa: E402
    Scene,
    box,
    cylinder,
    import_prop,
    kelvin_tint,
    key_x,
    look_at_rotation,
)

BUILDERS = {"jar_line": jar_line.build, "carton_packing": carton_packing.build}
BELT_LENGTH = 7.0


def main() -> None:
    spec_path, output, preview = sys.argv[sys.argv.index("--") + 1 :]
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    scene = Scene(spec)
    _render_settings(spec, output, preview == "1")
    _world(scene)
    _camera(spec)
    _lights(spec)
    _floor(scene)
    _belt(scene)
    _background(scene)
    BUILDERS[spec["builder"]](scene)
    if preview == "1":
        bpy.ops.render.render(write_still=True)
    else:
        bpy.ops.render.render(animation=True)


def _render_settings(spec, output: str, preview: bool) -> None:
    render = spec["render"]
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = render["samples"]
    scene.eevee.use_raytracing = False
    scene.render.resolution_x = render["width"]
    scene.render.resolution_y = render["height"]
    scene.render.fps = render["fps"]
    scene.frame_start = 0
    scene.frame_end = render["frames"] - 1
    scene.frame_current = int(render.get("preview_frame", 0))
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = render.get("exposure", 0.0)
    scene.render.filepath = output
    settings = scene.render.image_settings
    if preview:
        settings.file_format = "PNG"
        return
    settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "PERC_LOSSLESS"
    scene.render.ffmpeg.gopsize = render["fps"]


def _world(scene: Scene) -> None:
    look = scene.spec["look"]
    world = bpy.data.worlds.new("world")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    background = nodes["Background"]
    environment = nodes.new("ShaderNodeTexEnvironment")
    environment.image = bpy.data.images.load(scene.spec["hdri"])
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value = (0, 0, look["hdri_rotation"])
    coords = nodes.new("ShaderNodeTexCoord")
    links.new(coords.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
    links.new(environment.outputs["Color"], background.inputs["Color"])
    background.inputs["Strength"].default_value = look["hdri_strength"]


def _camera(spec) -> None:
    camera = spec["camera"]
    data = bpy.data.cameras.new("camera")
    data.lens = camera["focal_mm"]
    data.sensor_width = 36.0
    data.sensor_fit = "HORIZONTAL"
    obj = bpy.data.objects.new("camera", data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = camera["position"]
    obj.rotation_euler = look_at_rotation(camera["position"], camera["look_at"])
    bpy.context.scene.camera = obj


def _lights(spec) -> None:
    look = spec["look"]
    colour = kelvin_tint(look["light_warmth"])
    for index, x in enumerate((-1.4, 0.0, 1.4)):
        data = bpy.data.lights.new(f"strip_{index}", "AREA")
        data.shape = "RECTANGLE"
        data.size, data.size_y = 1.2, 0.25
        data.energy = 150 * look["light_energy"]
        data.color = colour
        obj = bpy.data.objects.new(f"strip_{index}", data)
        obj.location = (x, -0.3 + 0.2 * index, 3.0)
        bpy.context.scene.collection.objects.link(obj)


def _floor(scene: Scene) -> None:
    box("floor", (24, 24, 0.02), (0, 0, -0.01), scene.textured("floor", 0.4, (0.3, 0.3, 0.3)))


def _belt(scene: Scene) -> None:
    belt = scene.spec["belt"]
    top, width = belt["top"], belt["width"]
    rubber = scene.textured("belt", 2.0, (0.35, 0.35, 0.37))
    box("belt", (BELT_LENGTH, width, 0.02), (0, 0, top - 0.01), rubber)
    # The belt texture scrolls with the belt, keyed at the same instants as the items.
    mapping = next(n for n in rubber.node_tree.nodes if n.type == "MAPPING")
    for t, distance in belt["keys"]:
        mapping.inputs["Location"].default_value[0] = -distance
        mapping.inputs["Location"].keyframe_insert(
            "default_value", index=0, frame=scene.frame(t)
        )
    steel = scene.textured("steel", 1.0, metallic=1.0)
    painted = scene.textured("painted", 1.0, tuple(scene.spec["frame_colour"]), grey=True)
    for side in (-1, 1):
        y = side * (width / 2 + 0.03)
        box(f"frame_{side}", (BELT_LENGTH, 0.05, 0.12), (0, y, top - 0.05), painted, bevel=0.005)
        for index in range(8):
            x = -BELT_LENGTH / 2 + 0.3 + index * (BELT_LENGTH - 0.6) / 7
            box(f"leg_{side}_{index}", (0.05, 0.05, top - 0.1), (x, y, (top - 0.1) / 2), painted)
    for end in (-1, 1):
        roller = cylinder(f"roller_{end}", 0.04, width + 0.04, (end * BELT_LENGTH / 2, 0, top - 0.04), steel)
        roller.rotation_euler = (1.5708, 0, 0)


def _background(scene: Scene) -> None:
    assets = scene.assets
    rng = random.Random(scene.spec["look"]["props_seed"])
    for index in range(3):
        x = -3.2 + index * 2.6 + rng.uniform(-0.3, 0.3)
        import_prop(assets["prop_SM_BulkStorageRack_A03_01"], (x, 2.6, 0), rng.uniform(-0.05, 0.05))
        for slot in range(3):
            if rng.random() < 0.75:
                prop = rng.choice(("Cardbox_A1", "Cardbox_B1", "Cardbox_C1", "SM_Crate_A01_Blue_01"))
                location = (x - 0.75 + slot * 0.75, 2.6 + rng.uniform(-0.1, 0.1), 0.12)
                import_prop(assets[f"prop_{prop}"], location, rng.uniform(-0.3, 0.3), 0.6)
    for index in range(2):
        x = rng.choice((-2.6, -1.6, 1.4, 2.4)) + rng.uniform(-0.2, 0.2)
        pallet = (x, 1.35 + index * 0.1, 0)
        import_prop(assets["prop_Pallet_A1"], pallet, rng.uniform(-0.2, 0.2))
        for level in range(rng.randint(1, 3)):
            prop = rng.choice(("Cardbox_A1", "Cardbox_B1"))
            import_prop(assets[f"prop_{prop}"], (x, pallet[1], 0.15 + level * 0.5), rng.uniform(-0.1, 0.1))


if __name__ == "__main__":
    main()
