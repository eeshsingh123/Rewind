from rewind.contracts import ScenarioConfig, TimeSpan
from rewind.contracts.scenario import PropValue
from sim.plan import VIEW_EXIT, ground_truth, plan_scene
from sim.scenario import SCENARIOS, load_scenario

S1 = SCENARIOS / "s1_jar_line.yaml"
STEADY: dict[str, PropValue] = {
    "speed_m_s": 0.1,
    "speed_jitter": 0,
    "interval_jitter": 0,
    "camera_jitter_m": 0,
    "focal_jitter_mm": 0,
    "jam_probability": 0,
}


def steady_s1(faults: dict[str, float], jams: tuple[TimeSpan, ...] = ()) -> ScenarioConfig:
    config = load_scenario(S1)
    params = {**config.params, **STEADY}
    return config.model_copy(update={"faults": faults, "jams": jams, "params": params})


def test_skipped_caps_and_a_jam_lead_to_their_violations():
    jam = TimeSpan(start=10.0, end=20.0)
    truth = ground_truth(plan_scene(steady_s1({"missing_cap": 1.0}, (jam,))))
    left = [i for i in truth.items if VIEW_EXIT in i.t_cross and i.t_cross[VIEW_EXIT] < 59]
    # The capper zone is 0.3 m long, so a jar spends 3 s in it unless the jam holds it there.
    in_capper_at_jam = [
        i for i in truth.items if i.t_cross.get("capper", 99) <= 10 < i.t_cross["capper"] + 3
    ]

    assert left
    assert in_capper_at_jam
    assert all("cap_and_label_on_exit" in i.expected_violations for i in left)
    for item in truth.items:
        assert ("capper_dwell" in item.expected_violations) == (item in in_capper_at_jam)
    assert [event.span for event in truth.events] == [jam]
