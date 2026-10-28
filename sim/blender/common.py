import math
from pathlib import Path

import bpy
import mathutils


class Scene:
    # The plan JSON plus the shared materials, so builders don't rebuild them.
    def __init__(self, spec: dict) -> None:
        self.spec = spec
        self.fps = spec["render"]["fps"]
        self.assets = spec["assets"]
        self.materials: dict[str, bpy.types.Material] = {}

    def frame(self, t: float) -> float:
        return t * self.fps

    def textured(self, role: str, scale=1.0, tint=(1.0, 1.0, 1.0), metallic=0.0, grey=False):
        name = f"{role}_{scale}_{tint}_{metallic}_{grey}"
        if name not in self.materials:
            self.materials[name] = textured_material(
                name, self.assets, role, scale, tint, metallic, grey
            )
        return self.materials[name]

    def flat(self, name: str, colour, roughness=0.5, metallic=0.0, coat=0.0, alpha=1.0):
        if name not in self.materials:
            self.materials[name] = flat_material(name, colour, roughness, metallic, coat, alpha)
        return self.materials[name]


def textured_material(name, assets, role, scale, tint, metallic, grey):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = nodes["Principled BSDF"]
    coords = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    links.new(coords.outputs["Object"], mapping.inputs["Vector"])
    colour = _image_node(nodes, assets[f"{role}_color"], "sRGB")
    rough = _image_node(nodes, assets[f"{role}_roughness"], "Non-Color")
    normal = _image_node(nodes, assets[f"{role}_normalgl"], "Non-Color")
    for node in (colour, rough, normal):
        # Box projection over object coordinates needs no UVs on the generated meshes.
        node.projection = "BOX"
        node.projection_blend = 0.2
        links.new(mapping.outputs["Vector"], node.inputs["Vector"])
    multiply = nodes.new("ShaderNodeMix")
    multiply.data_type = "RGBA"
    multiply.blend_type = "MULTIPLY"
    multiply.inputs["Factor"].default_value = 1.0
    if grey:
        # Keeps the wear and grain of the texture but lets the tint choose the paint colour.
        to_grey = nodes.new("ShaderNodeRGBToBW")
        links.new(colour.outputs["Color"], to_grey.inputs["Color"])
        links.new(to_grey.outputs["Val"], multiply.inputs["A"])
    else:
        links.new(colour.outputs["Color"], multiply.inputs["A"])
    multiply.inputs["B"].default_value = (*tint, 1.0)
    links.new(multiply.outputs["Result"], shader.inputs["Base Color"])
    links.new(rough.outputs["Color"], shader.inputs["Roughness"])
    normal_map = nodes.new("ShaderNodeNormalMap")
    links.new(normal.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], shader.inputs["Normal"])
    shader.inputs["Metallic"].default_value = metallic
    return material


def flat_material(name, colour, roughness, metallic, coat, alpha):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = material.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (*colour, 1.0)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Coat Weight"].default_value = coat
    if alpha < 1.0:
        shader.inputs["Alpha"].default_value = alpha
        material.surface_render_method = "BLENDED"
    return material


def image_material(name: str, path: str, roughness=0.45):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = nodes["Principled BSDF"]
    image = _image_node(nodes, path, "sRGB")
    image.extension = "CLIP"
    links.new(image.outputs["Color"], shader.inputs["Base Color"])
    shader.inputs["Roughness"].default_value = roughness
    return material


def prop_material(name: str, folder: Path):
    # SimReady materials are MDL, which Blender can't read; bind their PNG maps directly.
    albedos = sorted(folder.rglob("*Albedo*.png"))
    if not albedos:
        return None
    words = set(name.lower().split("_"))
    albedo = max(albedos, key=lambda p: len(words & set(p.stem.lower().split("_"))))
    material = bpy.data.materials.new(f"prop_{name}")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    shader = nodes["Principled BSDF"]
    links.new(_image_node(nodes, str(albedo), "sRGB").outputs["Color"], shader.inputs["Base Color"])
    normal_path = albedo.with_name(albedo.name.replace("Albedo", "Normal"))
    if normal_path.exists():
        normal_map = nodes.new("ShaderNodeNormalMap")
        normal = _image_node(nodes, str(normal_path), "Non-Color")
        links.new(normal.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], shader.inputs["Normal"])
    shader.inputs["Roughness"].default_value = 0.7
    return material


def _image_node(nodes, path: str, colorspace: str):
    node = nodes.new("ShaderNodeTexImage")
    node.image = bpy.data.images.load(path, check_existing=True)
    node.image.colorspace_settings.name = colorspace
    return node


def box(name, size, location=(0, 0, 0), material=None, parent=None, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = size
    bpy.ops.object.transform_apply(scale=True)
    if bevel > 0:
        modifier = obj.modifiers.new("bevel", "BEVEL")
        modifier.width = bevel
        modifier.segments = 2
    return _finish(obj, material, parent)


def cylinder(
    name, radius, depth, location=(0, 0, 0), material=None, parent=None, vertices=48, caps=True
):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices,
        radius=radius,
        depth=depth,
        location=location,
        end_fill_type="NGON" if caps else "NOTHING",
    )
    obj = bpy.context.active_object
    obj.name = name
    bpy.ops.object.shade_smooth()
    return _finish(obj, material, parent)


def _finish(obj, material, parent):
    if material is not None:
        obj.data.materials.append(material)
    if parent is not None:
        obj.parent = parent
    return obj


def wrap_uv(obj) -> None:
    # Unrolls a sleeve so the middle of its image faces -y (the camera) and the seam faces +y.
    mesh = obj.data
    uv = mesh.uv_layers.active.data
    heights = [v.co.z for v in mesh.vertices]
    low, high = min(heights), max(heights)
    for polygon in mesh.polygons:
        coords = []
        for index in polygon.loop_indices:
            co = mesh.vertices[mesh.loops[index].vertex_index].co
            u = ((math.atan2(co.y, co.x) + math.pi / 2) / (2 * math.pi)) % 1.0
            coords.append((index, u, (co.z - low) / (high - low)))
        if max(c[1] for c in coords) - min(c[1] for c in coords) > 0.5:
            coords = [(i, u + 1.0 if u < 0.5 else u, v) for i, u, v in coords]
        for index, u, v in coords:
            uv[index].uv = (1.0 - u, v)


def empty(name, location=(0, 0, 0)):
    obj = bpy.data.objects.new(name, None)
    obj.location = location
    bpy.context.scene.collection.objects.link(obj)
    return obj


def import_prop(path: str, location, rotation_z=0.0, scale=1.0):
    before = set(bpy.context.scene.objects)
    bpy.ops.wm.usd_import(filepath=path, set_frame_range=False)
    added = [o for o in bpy.context.scene.objects if o not in before]
    root = empty(f"prop_{Path(path).stem}", location)
    root.rotation_euler = (0, 0, rotation_z)
    root.scale = (scale, scale, scale)
    folder = Path(path).parent
    for obj in added:
        if obj.parent is None:
            obj.parent = root
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            if slot.material is not None:
                replacement = prop_material(slot.material.name, folder)
                if replacement is not None:
                    slot.material = replacement
    return root


def key_x(scene: Scene, obj, offset: float) -> None:
    # The belt moves linearly between its keys, so keying the same instants moves items with it.
    for t, distance in scene.spec["belt"]["keys"]:
        obj.location.x = offset + distance
        obj.keyframe_insert("location", index=0, frame=scene.frame(t))


def key_pop(scene: Scene, obj, appears: float | None, ramp_s: float = 0.2) -> None:
    # Grows the part into place over ramp_s, ending when it is applied; None hides it for good.
    if appears is None:
        obj.hide_render = True
        obj.hide_viewport = True
        return
    if appears <= 0:
        return
    target = tuple(obj.scale)
    obj.scale = (0.0, 0.0, 0.0)
    obj.keyframe_insert("scale", frame=scene.frame(max(0.0, appears - ramp_s)))
    obj.scale = target
    obj.keyframe_insert("scale", frame=scene.frame(appears))


def look_at_rotation(position, target):
    direction = mathutils.Vector(target) - mathutils.Vector(position)
    return direction.to_track_quat("-Z", "Y").to_euler()


def kelvin_tint(warmth: float):
    # -1 cool white, +1 warm white.
    return (1.0, 0.93 - 0.05 * warmth, 0.86 - 0.12 * warmth) if warmth > 0 else (
        0.92 + 0.08 * (1 + warmth),
        0.95,
        1.0,
    )


def radians(degrees: float) -> float:
    return math.radians(degrees)
