import json
import pytest

from src import pipeline
from src.cache import cache_key, get_precomputed, put_precomputed
from src.pipeline import clean_and_parse_json

ABSTRACT = (
    "Artificial riprap crevices significantly increased juvenile round goby survival and local density by 340% "
    "compared to natural gravel-sand banks (p < 0.001). "
    "Round goby dominance occurred exclusively in low-flow reaches (< 0.25 m/s) where summer water temperatures exceeded 18.5°C. "
    "In high-flow riffles (> 0.65 m/s) native Cottus gobio maintained competitive dominance."
)


class _Msg:
    def __init__(self, content): self.content = content


class _Choice:
    def __init__(self, content): self.message = _Msg(content)


class _Resp:
    def __init__(self, content): self.choices = [_Choice(content)]


class FakeClient:
    """Returns queued replies in order and records every call."""
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Resp(self.replies.pop(0))


def _extraction(quote2):
    return json.dumps({
        "paper_id": "WRONG", "paper_title": "WRONG",
        "primary_hypothesis_name": "Habitat Facilitation Hypothesis",
        "hypothesis_summary": "Riprap boosts gobies under low flow.",
        "causal_edges": [
            {"edge_id": "E1", "cause_variable": "Artificial Riprap", "effect_variable": "Round Goby Density",
             "causal_direction": "Positive (+)", "mechanism_type": "Causal / Experimental", "uncertainty_level": "Low",
             "context": {"water_temperature": "> 18.5°C", "flow_velocity": "< 0.25 m/s",
                         "habitat_structure": "Riprap boulder crevices", "additional_moderators": None},
             "provenance_quote": "Artificial riprap crevices significantly increased juvenile round goby survival and local density by 340%"},
            {"edge_id": "E2", "cause_variable": "High Flow", "effect_variable": "Native Bullhead Dominance",
             "causal_direction": "Positive (+)", "mechanism_type": "Observational / Correlational", "uncertainty_level": "Moderate",
             "context": {"water_temperature": "Unspecified", "flow_velocity": "> 0.65 m/s",
                         "habitat_structure": "Unspecified", "additional_moderators": None},
             "provenance_quote": quote2},
        ],
        "orkg_triples": [{"subject": "Artificial Riprap", "predicate": "increases_density_of",
                          "object": "Round Goby", "context_qualifier": "flow < 0.25 m/s", "provenance_source": "P1"}],
    })


JUDGE = json.dumps({"causal_rigor_score": 0.7, "contextual_completeness": 0.6, "judge_rationale": "E2 label is debatable."})


def test_extraction_pipeline_measures_provenance_in_code_and_overrides_ids():
    client = FakeClient([_extraction("Native bullheads thrive wherever flow is strong in every river."), JUDGE])
    result = pipeline.run_extraction(client, "P1", "Title", ABSTRACT, "m", "j")
    assert result["paper_id"] == "P1" and result["paper_title"] == "Title"   # echoed ids are not trusted
    ev = result["evaluation"]
    assert ev["provenance_edges_exact"] == 1 and ev["provenance_edges_unverified"] == 1
    assert ev["provenance_faithfulness"] == 0.5
    assert ev["causal_rigor_score"] == 0.7 and ev["contextual_completeness"] == 0.6
    assert result["graph_diagnostics"]["is_dag"] is True
    assert len(client.calls) == 2
    assert client.calls[0]["model"] == "m" and client.calls[1]["model"] == "j"   # separate judge call


def test_judge_cannot_change_provenance_score():
    judge = json.dumps({"causal_rigor_score": 1.0, "contextual_completeness": 1.0,
                        "judge_rationale": "Everything is perfectly grounded.", "provenance_faithfulness": 1.0})
    client = FakeClient([_extraction("Made up quote that is not in the text at all whatsoever."), judge])
    result = pipeline.run_extraction(client, "P1", "T", ABSTRACT)
    assert result["evaluation"]["provenance_faithfulness"] == 0.5


def test_judge_prompt_contains_no_example_scores():
    assert "0.96" not in pipeline.JUDGE_SYSTEM_PROMPT and "0.96" not in pipeline.EXTRACTION_SYSTEM_PROMPT


def test_judge_failure_degrades_gracefully():
    client = FakeClient([_extraction("Native bullheads thrive wherever flow is strong."), "not json", "still not json"])
    result = pipeline.run_extraction(client, "P1", "T", ABSTRACT)
    ev = result["evaluation"]
    assert ev["causal_rigor_score"] is None and "unavailable" in ev["judge_rationale"]
    assert ev["provenance_edges_total"] == 2


def test_extraction_retries_once_on_invalid_json():
    client = FakeClient(["```oops```", _extraction("In high-flow riffles (> 0.65 m/s) native Cottus gobio maintained competitive dominance"), JUDGE])
    result = pipeline.run_extraction(client, "P1", "T", ABSTRACT)
    assert result["evaluation"]["provenance_faithfulness"] == 1.0
    assert len(client.calls) == 3


def test_extraction_raises_after_repeated_invalid_output():
    with pytest.raises(ValueError):
        pipeline.run_extraction(FakeClient(["nope", "still nope"]), "P1", "T", ABSTRACT)


def _extracted():
    client = FakeClient([_extraction("In high-flow riffles (> 0.65 m/s) native Cottus gobio maintained competitive dominance"), JUDGE])
    return pipeline.run_extraction(client, "P1", "T", ABSTRACT)


def test_simulation_adds_gates_and_flags_conflicts():
    extraction = _extracted()
    sim_reply = json.dumps({
        "hypothesis_status": "Partially Attenuated", "predicted_invasion_risk": "Moderate Risk",
        "causal_chain_reasoning": "Flow restoration removes the low-flow condition.",
        "edge_by_edge_impact": [
            {"edge_id": "E1", "pathway": "Artificial Riprap -> Round Goby Density",
             "status_under_intervention": "Active / Amplified", "explanation": "Riprap is retained."},
        ],
        "management_recommendation": "Restore lotic flow.",
    })
    client = FakeClient([sim_reply])
    sim = pipeline.run_simulation(
        client, extraction,
        "Moderate Warming (+2°C / > 19°C)",
        "Restore High Lotic Flow (> 0.6 m/s via Dam Removal)",
        "Artificial Riprap Boulders Retained",
        "Moderate Agricultural Runoff",
    )
    rows = {r["edge_id"]: r for r in sim["edge_by_edge_impact"]}
    assert set(rows) == {"E1", "E2"}                                   # every edge present, even if the LLM skipped one
    assert rows["E1"]["consistency"] == "Conflicts with context gate"   # flow 0.6+ violates "< 0.25 m/s"
    assert rows["E2"]["status_under_intervention"] == "Not returned by model"
    assert sim["gate_conflicts"] == 1
    assert "DETERMINISTIC CONTEXT GATES" in client.calls[0]["messages"][1]["content"]


def test_deterministic_fallback_has_no_llm_fields():
    sim = pipeline.deterministic_only_simulation(
        _extracted(), "Cool Shaded Regime (< 15°C)", "Low-Flow / Impounded Lentic (< 0.2 m/s)",
        "Mixed Bank Structure", "Moderate Agricultural Runoff",
    )
    assert "LLM unavailable" in sim["hypothesis_status"]
    assert all("context_gate" in r for r in sim["edge_by_edge_impact"])


def test_clean_and_parse_json_variants():
    assert clean_and_parse_json('<think>hmm</think>{"a": 1}') == {"a": 1}
    assert clean_and_parse_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert clean_and_parse_json('Here you go: {"a": 3} thanks') == {"a": 3}
    with pytest.raises(json.JSONDecodeError):
        clean_and_parse_json("no json here")


def test_precomputed_cache_round_trip(tmp_path):
    path = tmp_path / "cache.json"
    assert get_precomputed("P1", ABSTRACT, path) is None
    put_precomputed("P1", ABSTRACT, {"paper_id": "P1", "model_name": "m", "judge_model": "j"}, path)
    hit = get_precomputed("P1", ABSTRACT, path)
    assert hit["result"]["source"] == "precomputed" and hit["meta"]["model"] == "m"
    assert get_precomputed("P1", ABSTRACT + " edited", path) is None        # edited text invalidates the entry
    assert cache_key("P1", ABSTRACT) != cache_key("P2", ABSTRACT)
