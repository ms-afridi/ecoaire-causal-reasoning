from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class EcologicalContext(BaseModel):
    water_temperature: str = Field(
        default="Unspecified",
        description="Water temperature regime or threshold (e.g., '> 18°C', 'Cold < 12°C', 'Unspecified')"
    )
    flow_velocity: str = Field(
        default="Unspecified",
        description="Hydrological flow condition (e.g., 'Low flow < 0.2 m/s', 'High turbulence', 'Lentic')"
    )
    habitat_structure: str = Field(
        default="Unspecified",
        description="Physical bank or substrate condition (e.g., 'Riprap rock bank', 'Natural gravel bed')"
    )
    additional_moderators: Optional[str] = Field(
        default=None,
        description="Other ecological moderators such as pH, turbidity, salinity, or predator presence"
    )


class CausalEdge(BaseModel):
    edge_id: str = Field(..., description="Unique identifier like E1, E2, E3")
    cause_variable: str = Field(..., description="Normalized canonical name of the driver/cause variable")
    effect_variable: str = Field(..., description="Normalized canonical name of the affected/outcome variable")
    causal_direction: Literal["Positive (+)", "Negative (-)", "No Significant Effect (0)"] = Field(
        ..., description="Direction of the causal or mechanistic relationship"
    )
    mechanism_type: Literal["Causal / Experimental", "Observational / Correlational", "Mechanistic Hypothesis"] = Field(
        ..., description="Epistemic nature of the relationship reported in the paper"
    )
    uncertainty_level: Literal["Low", "Moderate", "High"] = Field(
        ..., description="Uncertainty based on p-values, sample constraints, or author hedging"
    )
    context: EcologicalContext = Field(
        ..., description="Boundary conditions under which this causal claim holds"
    )
    provenance_quote: str = Field(
        ..., description="Exact verbatim sentence or clause from the source text supporting this link"
    )
    provenance_status: Optional[Literal["exact", "snapped", "unverified"]] = Field(
        default=None,
        description=(
            "Set programmatically (not by the LLM): 'exact' = quote found verbatim in the source; "
            "'snapped' = quote replaced by the closest source sentence; 'unverified' = no match found"
        ),
    )
    provenance_similarity: Optional[float] = Field(
        default=None, description="Word-level similarity between the model's quote and the matched source text"
    )


class ORKGTriple(BaseModel):
    subject: str = Field(..., description="Subject entity in Open Research Knowledge Graph format")
    predicate: str = Field(..., description="Semantic predicate (e.g., 'positively_influences', 'inhibits_under_context')")
    object: str = Field(..., description="Object entity in Open Research Knowledge Graph format")
    context_qualifier: str = Field(..., description="Contextual condition qualifying this triple")
    provenance_source: str = Field(..., description="Paper ID + verbatim snippet reference")


class JudgeOutput(BaseModel):
    """What the separate LLM-judge call returns. Provenance is NOT judged by the LLM."""
    causal_rigor_score: float = Field(..., ge=0.0, le=1.0)
    contextual_completeness: float = Field(..., ge=0.0, le=1.0)
    judge_rationale: str = Field(...)


class EvaluationMetrics(BaseModel):
    provenance_faithfulness: float = Field(
        ..., ge=0.0, le=1.0,
        description="Computed in code: share of edges whose quote is found verbatim in the source text",
    )
    causal_rigor_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="0.0 to 1.0 LLM-judge score: correlation vs causation, direction, uncertainty",
    )
    contextual_completeness: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="0.0 to 1.0 LLM-judge score: capture of environmental moderators stated in the text",
    )
    judge_rationale: str = Field(
        default="", description="Concise critique from the LLM judge explaining the scores and error severity"
    )
    provenance_edges_exact: int = 0
    provenance_edges_snapped: int = 0
    provenance_edges_unverified: int = 0
    provenance_edges_total: int = 0
    evaluation_note: str = Field(
        default=(
            "provenance_faithfulness is computed programmatically. causal_rigor_score and "
            "contextual_completeness come from an LLM judge and are not validated against human annotations."
        )
    )


class ExtractionCore(BaseModel):
    """The part of the result that the extraction LLM call must return."""
    paper_id: str = Field(..., description="Identifier of the analyzed paper")
    paper_title: str = Field(..., description="Title of the analyzed paper")
    primary_hypothesis_name: str = Field(
        ..., description="Canonical name of the ecological hypothesis (e.g., 'Enemy Release Hypothesis', 'Biotic Resistance Hypothesis')"
    )
    hypothesis_summary: str = Field(
        ..., description="1-2 sentence synthesis of the core scientific claim and its boundary conditions"
    )
    causal_edges: List[CausalEdge] = Field(
        ..., description="List of structured causal relationships extracted from the text"
    )
    orkg_triples: List[ORKGTriple] = Field(
        ..., description="Machine-actionable Semantic Web triples following the ORKG data model"
    )


class HypothesisExtractionResult(ExtractionCore):
    evaluation: EvaluationMetrics = Field(
        ..., description="Programmatic provenance check + separate LLM-as-a-Judge evaluation"
    )
    graph_diagnostics: Optional[Dict[str, Any]] = Field(
        default=None, description="Deterministic graph checks (DAG test, path signs)"
    )
    model_name: Optional[str] = None
    judge_model: Optional[str] = None
    source: str = Field(default="live", description="'live' or 'precomputed'")


class EdgeImpact(BaseModel):
    edge_id: str
    pathway: str = ""
    status_under_intervention: str
    explanation: str = ""


class SimulationResult(BaseModel):
    hypothesis_status: str
    predicted_invasion_risk: str
    causal_chain_reasoning: str
    edge_by_edge_impact: List[EdgeImpact]
    management_recommendation: str
