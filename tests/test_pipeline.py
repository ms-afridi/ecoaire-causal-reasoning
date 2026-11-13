import pytest
from src.schema import (
    EcologicalContext,
    CausalEdge,
    ORKGTriple,
    EvaluationMetrics,
    HypothesisExtractionResult,
)
from src.graph_builder import build_causal_networkx_graph
from src.report_generator import generate_markdown_report, generate_orkg_jsonld_export


def get_mock_extraction_payload() -> dict:
    return {
        "paper_id": "TEST_01",
        "paper_title": "Test Riprap Facilitation of Invasive Round Goby",
        "primary_hypothesis_name": "Habitat Facilitation Hypothesis",
        "hypothesis_summary": "Artificial riprap increases juvenile round goby density under low flow.",
        "causal_edges": [
            {
                "edge_id": "E1",
                "cause_variable": "Artificial Riprap Bank",
                "effect_variable": "Round Goby Density",
                "causal_direction": "Positive (+)",
                "mechanism_type": "Causal / Experimental",
                "uncertainty_level": "Low",
                "context": {
                    "water_temperature": "> 18.5°C",
                    "flow_velocity": "< 0.25 m/s",
                    "habitat_structure": "Riprap boulder crevices",
                    "additional_moderators": None,
                },
                "provenance_quote": "Artificial riprap crevices significantly increased juvenile round goby survival and local density by 340%.",
            }
        ],
        "orkg_triples": [
            {
                "subject": "Artificial Riprap Bank",
                "predicate": "increases_density_of",
                "object": "Round Goby (Neogobius melanostomus)",
                "context_qualifier": "flow < 0.25 m/s, temp > 18.5C",
                "provenance_source": "TEST_01",
            }
        ],
        "evaluation": {
            "provenance_faithfulness": 0.98,
            "causal_rigor_score": 0.95,
            "contextual_completeness": 0.96,
            "judge_rationale": "All extracted links are grounded in verbatim experimental quotes.",
        },
    }


def test_pydantic_schema_validation():
    payload = get_mock_extraction_payload()
    model = HypothesisExtractionResult(**payload)
    assert model.paper_id == "TEST_01"
    assert len(model.causal_edges) == 1
    assert model.causal_edges[0].causal_direction == "Positive (+)"
    assert model.evaluation.provenance_faithfulness >= 0.9


def test_causal_graph_construction():
    payload = get_mock_extraction_payload()
    G = build_causal_networkx_graph(payload["causal_edges"])
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1
    assert G.has_edge("Artificial Riprap Bank", "Round Goby Density")


def test_report_generation():
    payload = get_mock_extraction_payload()
    md_report = generate_markdown_report(payload)
    jsonld_report = generate_orkg_jsonld_export(payload)
    assert "Habitat Facilitation Hypothesis" in md_report
    assert "ecoaire:ScientificHypothesisAssessment" in jsonld_report


def test_schema_rejects_out_of_range_scores():
    with pytest.raises(Exception):
        EvaluationMetrics(provenance_faithfulness=1.4)


def test_judge_scores_are_optional_when_judge_unavailable():
    ev = EvaluationMetrics(provenance_faithfulness=0.5)
    assert ev.causal_rigor_score is None
    md = generate_markdown_report({"evaluation": ev.model_dump(), "causal_edges": [], "orkg_triples": []})
    assert "n/a" in md
