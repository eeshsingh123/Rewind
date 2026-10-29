import argparse
import json
import random
import subprocess
import time
from pathlib import Path

from pydantic import TypeAdapter
from pydantic_settings import BaseSettings, SettingsConfigDict

from rewind.contracts import Zone
from sim.assets import fetch_assets
from sim.labels import jar_label, shipping_label
from sim.overlay import draw_preview
from sim.plan import ScenePlan, ground_truth, plan_scene
from sim.scenario import load_scenario

BUILD_SCRIPT = Path(__file__).parent / "blender" / "build_scene.py"
FRAME_COLOURS = ((0.25, 0.3, 0.35), (0.12, 0.25, 0.45), (0.3, 0.3, 0.3))
MACHINE_COLOURS = ((0.85, 0.6, 0.08), (0.1, 0.3, 0.6), (0.7, 0.12, 0.08), (0.2, 0.45, 0.25))


class SimSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REWIND_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    blender_path: Path = Path("C:/Program Files/Blender Foundation/Blender 4.5/blender.exe")
    data_root: Path = Path("data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a scenario with its answer key.")
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--duration", type=float, help="shorter footage, for quick checks")
    parser.add_argument("--still", type=float, help="render one frame at this time, as a PNG")
    args = parser.parse_args()
    settings = SimSettings()
    config = load_scenario(args.scenario, args.seed)
    if args.duration is not None:
        config = config.model_copy(update={"duration_s": args.duration})
    plan = plan_scene(config)
    out = args.out or settings.data_root / "sim" / f"{config.name}-seed{config.seed}"
    out.mkdir(parents=True, exist_ok=True)
    timings = render(plan, out, settings, args.still)
    print(json.dumps({"out": str(out.resolve()), "seconds": timings}, indent=2))


def render(
    plan: ScenePlan, out: Path, settings: SimSettings, still: float | None
) -> dict[str, float]:
    timings: dict[str, float] = {}
    started = time.monotonic()
    truth = ground_truth(plan)
    (out / "ground_truth.json").write_text(truth.model_dump_json(indent=2), encoding="utf-8")
    zones = TypeAdapter(tuple[Zone, ...]).dump_json(truth.zones, indent=2)
    (out / "zones.json").write_bytes(zones)
    assets = fetch_assets(settings.data_root / "sim-assets")
    spec = blender_spec(plan, assets, out / "textures", still)
    (out / "scene.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    timings["prepare"] = time.monotonic() - started
    started = time.monotonic()
    target = out / ("still.png" if still is not None else "clean_")
    _blender(settings.blender_path, out / "scene.json", target, still is not None)
    timings["blender"] = time.monotonic() - started
    if still is not None:
        return timings
    clean = out / "clean.mp4"
    produced = sorted(out.glob("clean_*.mp4"))
    produced[-1].replace(clean)
    started = time.monotonic()
    degrade(plan, clean, out / "video.mp4")
    timings["degrade"] = time.monotonic() - started
    started = time.monotonic()
    draw_preview(plan, out / "video.mp4", out / "preview.mp4")
    timings["preview"] = time.monotonic() - started
    (out / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    return timings


def blender_spec(
    plan: ScenePlan, assets: dict[str, str], textures: Path, still: float | None
) -> dict[str, object]:
    config, line, look = plan.config, plan.line, plan.look
    textures.mkdir(exist_ok=True)
    rng = random.Random(look.props_seed)
    hdri = str(config.params.get("hdri", "empty_warehouse_01"))
    items: list[dict[str, object]] = []
    for item in plan.items:
        entry: dict[str, object] = {
            "id": item.id,
            "offset": item.offset,
            "parts": item.timeline.parts,
            "station_times": item.station_times,
            "style": item.style,
        }
        if line.item_kind == "jar":
            design, contents = int(item.style["label_design"]), str(item.style["contents"])
            name = f"jar_{design}_{contents.replace(' ', '_')}.png"
            entry["label_image"] = str(jar_label(textures / name, design, contents))
        elif item.order_id is not None:
            label_rng = random.Random(f"{config.seed}:{item.order_id}")
            with_qr = textures / f"ship_{item.order_id}.png"
            plain = textures / f"ship_{item.order_id}_plain.png"
            entry["label_image"] = str(shipping_label(with_qr, item.order_id, True, label_rng))
            label_rng = random.Random(f"{config.seed}:{item.order_id}")
            entry["label_plain"] = str(shipping_label(plain, item.order_id, False, label_rng))
        items.append(entry)
    return {
        "builder": config.builder,
        "render": {
            "width": config.width,
            "height": config.height,
            "fps": config.fps,
            "frames": round(config.duration_s * config.fps),
            "samples": int(config.params.get("render_samples", 12)),
            "preview_frame": 0 if still is None else round(still * config.fps),
        },
        "camera": plan.camera.model_dump(),
        "look": {
            "hdri_strength": look.hdri_strength,
            "hdri_rotation": look.hdri_rotation,
            "light_energy": look.light_energy,
            "light_warmth": look.light_warmth,
            "props_seed": look.props_seed,
        },
        "hdri": assets[f"hdri_{hdri}"],
        "assets": assets,
        "frame_colour": rng.choice(FRAME_COLOURS),
        "machine_colour": rng.choice(MACHINE_COLOURS),
        "belt": {
            "top": line.belt_top,
            "width": line.belt_width,
            "speed": plan.belt.speed,
            "keys": plan.belt.keys(config.duration_s),
        },
        "item_size": line.item_size,
        "stations": [
            {
                "key": station.key,
                "start": station.start,
                "end": station.end,
                "times": sorted(i.station_times[station.key] for i in plan.items),
            }
            for station in line.stations
        ],
        "items": items,
    }


def degrade(plan: ScenePlan, clean: Path, output: Path) -> None:
    # CCTV look: slight shake, softness, sensor noise, flatter colour and heavy H.264.
    look, config = plan.look, plan.config
    shake = look.shake_px
    margin = int(shake * 2) + 2
    crop = (
        f"crop=iw-{2 * margin}:ih-{2 * margin}:"
        f"{margin}+{shake:.2f}*sin(2*PI*t*0.7)+{shake / 2:.2f}*sin(2*PI*t*3.1):"
        f"{margin}+{shake:.2f}*sin(2*PI*t*0.5+1)+{shake / 2:.2f}*sin(2*PI*t*2.3)"
    )
    filters = ",".join(
        (
            crop,
            f"scale={config.width}:{config.height}",
            f"gblur=sigma={look.blur:.2f}",
            f"noise=alls={look.noise:.0f}:allf=t",
            "eq=saturation=0.85:contrast=0.95",
        )
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(clean), "-vf", filters,
            "-c:v", "libx264", "-preset", "medium", "-crf", str(look.crf),
            "-g", str(config.fps * 2), "-pix_fmt", "yuv420p", "-an", str(output),
        ],
        check=True,
    )  # fmt: skip


def _blender(blender: Path, spec: Path, target: Path, still: bool) -> None:
    command = [
        str(blender), "-b", "--factory-startup", "--python-exit-code", "1",
        "--python", str(BUILD_SCRIPT), "--", str(spec), str(target), "1" if still else "0",
    ]  # fmt: skip
    log = target.parent / "blender.log"
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Blender failed (exit {result.returncode}); see {log}")


if __name__ == "__main__":
    main()
