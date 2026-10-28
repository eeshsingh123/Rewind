import random

from rewind.contracts.scenario import PropValue
from sim.layout import ItemTimeline, Line, Part, Station

CAPPER_MAX_S = 5.0

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


def judge(item: ItemTimeline) -> tuple[str, ...]:
    found: list[str] = []
    capped_and_labelled = item.has("cap", item.last_seen) and item.has("label", item.last_seen)
    if item.exits and not capped_and_labelled:
        found.append("cap_and_label_on_exit")
    if item.zone_stay.get("capper", 0.0) > CAPPER_MAX_S:
        found.append("capper_dwell")
    return tuple(found)


def style(rng: random.Random) -> dict[str, PropValue]:
    return {
        "contents": rng.choice(sorted(CONTENTS)),
        "cap_colour": rng.choice(sorted(CAP_COLOURS)),
        "label_design": rng.randrange(4),
    }


JAR_LINE = Line(
    item_kind="jar",
    item_size=(0.085, 0.085, 0.13),
    belt_top=0.82,
    belt_width=0.22,
    stations=(
        Station(key="capper", start=-0.45, end=-0.15, parts=(Part("cap", "missing_cap"),)),
        Station(key="labeller", start=0.2, end=0.5, parts=(Part("label", "missing_label"),)),
    ),
    judge=judge,
    style=style,
)
