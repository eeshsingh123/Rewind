import io
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

# CC0 ambientCG materials, by the role they play in the scene.
TEXTURES = {
    "cardboard": "Cardboard004",
    "belt": "Rubber004",
    "steel": "Metal032",
    "painted": "PaintedMetal004",
    "floor": "Concrete034",
}
TEXTURE_MAPS = ("Color", "NormalGL", "Roughness")
# CC0 Poly Haven HDRIs; a scenario picks one with its `hdri` param.
HDRIS = ("empty_warehouse_01", "industrial_workshop_foundry")
# CC-BY-4.0 SimReady props, chosen because each is one USD file with plain (non-UDIM) textures.
SIMREADY_REPO = "nvidia/PhysicalAI-SimReady-Warehouse-01"
SIMREADY_PROPS = (
    "Pallet_A1",
    "Cardbox_A1",
    "Cardbox_B1",
    "Cardbox_C1",
    "SM_Crate_A01_Blue_01",
    "SM_BulkStorageRack_A03_01",
)
USER_AGENT = "rewind-sim/0.1"


def fetch_assets(cache: Path) -> dict[str, str]:
    # Downloads only what is missing; returns asset name -> local path for the Blender build.
    paths: dict[str, str] = {}
    for role, asset_id in TEXTURES.items():
        folder = _texture(cache / "textures" / asset_id, asset_id)
        for name in TEXTURE_MAPS:
            paths[f"{role}_{name.lower()}"] = str(folder / f"{asset_id}_1K-JPG_{name}.jpg")
    for hdri in HDRIS:
        paths[f"hdri_{hdri}"] = str(_hdri(cache / "hdri", hdri))
    simready = _simready(cache / "simready")
    for prop in SIMREADY_PROPS:
        paths[f"prop_{prop}"] = str(simready / "Props" / "general" / prop / f"{prop}.usd")
    return paths


def _texture(folder: Path, asset_id: str) -> Path:
    if all((folder / f"{asset_id}_1K-JPG_{name}.jpg").exists() for name in TEXTURE_MAPS):
        return folder
    archive = _download(f"https://ambientcg.com/get?file={asset_id}_1K-JPG.zip")
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        for name in TEXTURE_MAPS:
            wanted = f"{asset_id}_1K-JPG_{name}.jpg"
            # Read by exact name and write by our own path, so archive paths never reach the disk.
            (folder / wanted).write_bytes(zipped.read(wanted))
    return folder


def _hdri(folder: Path, hdri: str) -> Path:
    path = folder / f"{hdri}_1k.hdr"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        url = f"https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/{hdri}_1k.hdr"
        path.write_bytes(_download(url))
    return path


def _simready(folder: Path) -> Path:
    patterns: list[str] = []
    for prop in SIMREADY_PROPS:
        base = f"Props/general/{prop}"
        patterns += [f"{base}/{prop}.usd", f"{base}/materials/*"]
    # The token, if needed, comes from HF_TOKEN in the environment.
    snapshot_download(SIMREADY_REPO, repo_type="dataset", allow_patterns=patterns, local_dir=folder)
    return folder


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()
