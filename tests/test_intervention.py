from src.intervention import (
    INF, parse_condition, relation, evaluate_context_gates, consistency_flag,
    TEMP_OPTIONS, FLOW_OPTIONS, BANK_OPTIONS, WATER_OPTIONS,
)


def test_parse_upper_lower_and_range():
    assert parse_condition("< 0.25 m/s", "flow") == (-INF, 0.25)
    assert parse_condition("> 18.5°C", "temperature") == (18.5, INF)
    assert parse_condition("Low flow < 0.2 m/s", "flow") == (-INF, 0.2)
    assert parse_condition("0.45–0.85 m/s", "flow") == (0.45, 0.85)
    assert parse_condition("16-18°C", "temperature") == (16.0, 18.0)


def test_parse_combined_bounds_intersect():
    assert parse_condition("> 18°C and < 25°C", "temperature") == (18.0, 25.0)


def test_parse_units_are_separate():
    text = "conductivity > 650 µS/cm; TP > 85 µg/L"
    assert parse_condition(text, "conductivity") == (650.0, INF)
    assert parse_condition(text, "phosphorus") == (85.0, INF)
    assert parse_condition(text, "flow") is None


def test_unparseable_text_is_none():
    assert parse_condition("Unspecified", "flow") is None
    assert parse_condition("Lentic", "flow") is None
    assert parse_condition(None, "temperature") is None


def test_relation_logic():
    assert relation((19, INF), (18.5, INF)) == "Satisfied"
    assert relation((-INF, 15), (18.5, INF)) == "Violated"
    assert relation((16, 20), (18.5, INF)) == "Partial"


EDGE = {
    "edge_id": "E1",
    "context": {
        "water_temperature": "> 18.5°C",
        "flow_velocity": "< 0.25 m/s",
        "habitat_structure": "Riprap boulder crevices",
        "additional_moderators": None,
    },
}


def _params(temp, flow, bank, water):
    return {"temp_shift": temp, "flow_regime": flow, "bank_restoration": bank, "water_quality_regime": water}


def test_gate_violated_when_intervention_leaves_reported_conditions():
    p = _params(
        "Cool Shaded Regime (< 15°C)",
        "Restore High Lotic Flow (> 0.6 m/s via Dam Removal)",
        "Remove Riprap -> Natural Gravel/Cobble & Canopy",
        "Moderate Agricultural Runoff",
    )
    gate = evaluate_context_gates([EDGE], p)["E1"]
    assert gate["summary"] == "Violated"
    assert gate["dimensions"] == {"temperature": "Violated", "flow": "Violated", "habitat": "Violated"}


def test_gate_satisfied_when_conditions_match():
    p = _params(
        "Moderate Warming (+2°C / > 19°C)",
        "Low-Flow / Impounded Lentic (< 0.2 m/s)",
        "Artificial Riprap Boulders Retained",
        "Moderate Agricultural Runoff",
    )
    assert evaluate_context_gates([EDGE], p)["E1"]["summary"] == "Satisfied"


def test_gate_not_testable_without_thresholds():
    edge = {"edge_id": "E9", "context": {"water_temperature": "Unspecified", "flow_velocity": "Lentic",
                                         "habitat_structure": "Unspecified", "additional_moderators": None}}
    p = _params(list(TEMP_OPTIONS)[0], list(FLOW_OPTIONS)[0], list(BANK_OPTIONS)[0], list(WATER_OPTIONS)[0])
    assert evaluate_context_gates([edge], p)["E9"]["summary"] == "Not testable"


def test_consistency_flag():
    assert consistency_flag("Active / Amplified", "Violated") == "Conflicts with context gate"
    assert consistency_flag("Blocked by Context Threshold", "Violated") == "Consistent"
    assert consistency_flag("Blocked by Context Threshold", "Satisfied") == "Conflicts with context gate"
    assert consistency_flag("Weakened", "Not testable") == "n/a"
