from src.graph_analysis import analyze_causal_graph


def _edge(eid, cause, effect, direction="Positive (+)", unc="Low"):
    return {"edge_id": eid, "cause_variable": cause, "effect_variable": effect,
            "causal_direction": direction, "uncertainty_level": unc,
            "mechanism_type": "Causal / Experimental", "context": {}, "provenance_quote": ""}


def test_path_sign_is_product_of_edge_signs():
    edges = [
        _edge("E1", "Riprap", "Goby Density", "Positive (+)", "Low"),
        _edge("E2", "Goby Density", "Native Bullhead", "Negative (-)", "High"),
    ]
    d = analyze_causal_graph(edges)
    assert d["is_dag"] is True
    assert d["root_drivers"] == ["Riprap"] and d["outcomes"] == ["Native Bullhead"]
    path = d["driver_to_outcome_paths"][0]
    assert path["net_effect"] == "-" and path["max_uncertainty"] == "High"
    assert path["edge_ids"] == ["E1", "E2"]


def test_double_negative_is_positive_and_null_edge_zeroes_path():
    pos = analyze_causal_graph([_edge("E1", "A", "B", "Negative (-)"), _edge("E2", "B", "C", "Negative (-)")])
    assert pos["driver_to_outcome_paths"][0]["net_effect"] == "+"
    zero = analyze_causal_graph([_edge("E1", "A", "B"), _edge("E2", "B", "C", "No Significant Effect (0)")])
    assert zero["driver_to_outcome_paths"][0]["net_effect"] == "0"


def test_cycle_is_reported_not_hidden():
    d = analyze_causal_graph([_edge("E1", "A", "B"), _edge("E2", "B", "A")])
    assert d["is_dag"] is False and d["cycles"]


def test_duplicate_cause_effect_pairs_are_flagged():
    d = analyze_causal_graph([_edge("E1", "A", "B"), _edge("E2", "A", "B", "Negative (-)")])
    assert d["duplicate_cause_effect_pairs"] == ["A -> B"]
