from pathlib import Path

import yaml

from rewind.contracts import ScenarioConfig

SCENARIOS = Path(__file__).parent / "scenarios"


def load_scenario(path: Path, seed: int | None = None) -> ScenarioConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if seed is not None:
        data["seed"] = seed
    return ScenarioConfig.model_validate(data)
