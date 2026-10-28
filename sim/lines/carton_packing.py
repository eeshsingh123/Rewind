import random

from rewind.contracts.scenario import PropValue
from sim.layout import ItemTimeline, Line, Part, Station

TAPE_COLOURS = {
    "brown": (0.55, 0.36, 0.16),
    "clear": (0.75, 0.68, 0.52),
    "white": (0.85, 0.85, 0.82),
}
FLAPS = ("front", "back", "left", "right")


def judge(item: ItemTimeline) -> tuple[str, ...]:
    found: list[str] = []
    entered_sealing = item.zone_entry.get("sealing")
    if entered_sealing is not None and not item.has("label", entered_sealing):
        found.append("label_at_sealing")
    if item.exits and not item.has("tape", item.last_seen):
        found.append("tape_on_exit")
    if item.exits and not item.has("flaps", item.last_seen):
        found.append("flaps_before_tape")
    if not (item.has("label", item.last_seen) and item.has("qr", item.last_seen)):
        found.append("shipping_label_qr_at_end")
    return tuple(found)


def style(rng: random.Random) -> dict[str, PropValue]:
    return {
        "cardboard_tint": round(rng.uniform(0.8, 1.1), 3),
        "tape_colour": rng.choice(sorted(TAPE_COLOURS)),
        # Which flap stays up if the folder fails on this carton.
        "open_flap": rng.choice(FLAPS),
    }


CARTON_PACKING = Line(
    item_kind="carton",
    item_size=(0.34, 0.26, 0.22),
    belt_top=0.78,
    belt_width=0.42,
    stations=(
        Station(key="folder", start=-1.0, end=-0.6, parts=(Part("flaps", "flap_up"),)),
        Station(
            key="labeller",
            start=-0.3,
            end=0.0,
            parts=(Part("label", "missing_label"), Part("qr", "missing_qr")),
        ),
        Station(key="sealing", start=0.35, end=0.75, parts=(Part("tape", "missing_tape"),)),
    ),
    judge=judge,
    style=style,
)
