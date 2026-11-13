import json
import os
import re
from typing import Any, Callable, Dict, List, Optional

from src.graph_analysis import analyze_causal_graph
from src.intervention import consistency_flag, evaluate_context_gates
from src.schema import (
    EvaluationMetrics,
    ExtractionCore,
    HypothesisExtractionResult,
    JudgeOutput,
    SimulationResult,
)
from src.verification import verify_edges

DEFAULT_MODEL = os.environ.get("ECOAIRE_MODEL", "openai/gpt-oss-120b")
JUDGE_MODEL = os.environ.get("ECOAIRE_JUDGE_MODEL", DEFAULT_MODEL)


def clean_and_parse_json(raw_content: str) -> Dict[str, Any]:
    """Extract JSON from a model reply that may contain <think> blocks or ```json fences."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_content or "", flags=re.DOTALL).strip()
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
        if match:
            cleaned = match.group(1)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def chat_json(
    client: Any,
    model: str,
    system: str,
    user: str,
    temperature: float,
    validate: Callable[[Dict[str, Any]], Dict[str, Any]],
    max_retries: int = 1,
) -> Dict[str, Any]:
    """Call the model, parse and validate JSON, and retry once on malformed output."""
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        prompt = user if attempt == 0 else (
            user + "\n\nYour previous reply was rejected by schema validation: "
            f"{str(last_error)[:600]}\n"
            "Return ONLY one valid JSON object that follows the schema exactly, using the exact allowed strings "
            "(for example causal_direction must be exactly 'Positive (+)', 'Negative (-)' or 'No Significant Effect (0)')."
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        try:
            return validate(clean_and_parse_json(response.choices[0].message.content))
        except Exception as exc:  # JSONDecodeError, pydantic ValidationError, ...
            last_error = exc
    raise ValueError(f"Model returned invalid output after {max_retries + 1} attempts: {last_error}")


# ----------------------------------------------------------------------------------------
# Tolerant normalisation of enum-like fields (models often answer '-' instead of 'Negative (-)')
# ----------------------------------------------------------------------------------------
def _canon_direction(value: Any) -> Any:
    t = str(value).strip().lower()
    if t.startswith("no ") or t in {"0", "none", "null", "neutral", "nil"}:
        return "No Significant Effect (0)"
    if t.startswith("pos") or t in {"+", "plus", "increase", "increases", "increasing"}:
        return "Positive (+)"
    if t.startswith("neg") or t in {"-", "\u2212", "\u2013", "minus", "decrease", "decreases", "decreasing"}:
        return "Negative (-)"
    return value


def _canon_mechanism(value: Any) -> Any:
    t = str(value).strip().lower()
    if "observ" in t or "correl" in t:
        return "Observational / Correlational"
    if "mechanis" in t or "hypothes" in t:
        return "Mechanistic Hypothesis"
    if "causal" in t or "experiment" in t:
        return "Causal / Experimental"
    return value


def _canon_uncertainty(value: Any) -> Any:
    t = str(value).strip().lower()
    if t.startswith("low"):
        return "Low"
    if t.startswith(("mod", "med", "mid")):
        return "Moderate"
    if t.startswith("high"):
        return "High"
    return value


def normalize_extraction(data: Dict[str, Any]) -> Dict[str, Any]:
    """Map common spelling variants onto the schema's exact strings. Unknown values are left
    untouched so that genuinely wrong output still fails validation (and triggers the retry)."""
    out = dict(data)
    edges = out.get("causal_edges")
    if isinstance(edges, list):
        fixed = []
        for edge in edges:
            if not isinstance(edge, dict):
                fixed.append(edge)
                continue
            e = dict(edge)
            if "causal_direction" in e:
                e["causal_direction"] = _canon_direction(e["causal_direction"])
            if "mechanism_type" in e:
                e["mechanism_type"] = _canon_mechanism(e["mechanism_type"])
            if "uncertainty_level" in e:
                e["uncertainty_level"] = _canon_uncertainty(e["uncertainty_level"])
            ctx = e.get("context")
            if isinstance(ctx, dict):
                ctx = dict(ctx)
                for key in ("water_temperature", "flow_velocity", "habitat_structure"):
                    if ctx.get(key) in (None, ""):
                        ctx[key] = "Unspecified"
                e["context"] = ctx
            fixed.append(e)
        out["causal_edges"] = fixed
    return out


# ----------------------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------------------
EXTRACTION_SYSTEM_PROMPT = """You are EcoAIRE-Agent, a research-prototype Natural Language Processing and Causal AI component for freshwater ecology and invasion biology. It follows the Open Research Knowledge Graph (ORKG) data model.
Your task is to analyze scientific text and extract structured, provenance-aware causal representations.

You MUST return ONLY valid JSON strictly adhering to this exact schema:
{
  "paper_id": "string",
  "paper_title": "string",
  "primary_hypothesis_name": "string (e.g., Habitat Facilitation Hypothesis, Biotic Resistance Hypothesis)",
  "hypothesis_summary": "string (1-2 sentence synthesis of the core claim and boundary conditions)",
  "causal_edges": [
    {
      "edge_id": "E1",
      "cause_variable": "Normalized concise variable name (max 5 words)",
      "effect_variable": "Normalized concise variable name (max 5 words)",
      "causal_direction": "Positive (+)" | "Negative (-)" | "No Significant Effect (0)",
      "mechanism_type": "Causal / Experimental" | "Observational / Correlational" | "Mechanistic Hypothesis",
      "uncertainty_level": "Low" | "Moderate" | "High",
      "context": {
        "water_temperature": "Specific threshold copied from the text, or 'Unspecified'",
        "flow_velocity": "Specific threshold copied from the text, or 'Unspecified'",
        "habitat_structure": "Specific substrate/bank condition or 'Unspecified'",
        "additional_moderators": "Other moderators with units as written (conductivity, nutrients, turbidity) or null"
      },
      "provenance_quote": "EXACT verbatim sentence or clause copied from the input text supporting this causal link"
    }
  ],
  "orkg_triples": [
    {
      "subject": "Canonical Subject Entity",
      "predicate": "orkg_predicate (e.g., increases_density_of, suppresses_abundance_of, triggers_outbreak_in)",
      "object": "Canonical Object Entity",
      "context_qualifier": "Env condition qualifier (e.g., flow < 0.25 m/s & temp > 18.5C)",
      "provenance_source": "Paper ID + exact clause"
    }
  ]
}

CRITICAL RULES:
0. Enum fields must use the EXACT strings shown in the schema, never abbreviations. For example causal_direction is "Negative (-)", never "-".
1. Extract between 3 and 5 distinct causal_edges that capture both the invasive species impact AND the protective/restoration boundary conditions. Extract only links that the text states; never add links from background knowledge.
2. Keep cause_variable and effect_variable concise (2 to 5 words) so nodes connect cleanly in a Directed Acyclic Graph (DAG).
3. provenance_quote MUST be copied character for character from the provided text. Do not paraphrase, shorten with ellipses, or merge sentences. If no sentence supports a link, do not output that link.
4. Copy numeric thresholds and units into the context fields exactly as written in the text.
5. Generate 3 to 5 ORKG semantic triples matching the causal links.
"""

JUDGE_SYSTEM_PROMPT = """You are an independent reviewer of automated scientific information extraction.
You receive a source text, the structured causal extraction produced from it, and the results of a programmatic quote check.
Quote grounding is already measured in code: do NOT score it. Score only the two dimensions below, each from 0.0 to 1.0.

1. causal_rigor_score: Are 'Causal / Experimental' labels used only where the text reports a manipulation or controlled/BACI design, and 'Observational / Correlational' otherwise? Are directions correct? Do uncertainty levels match the reported p-values or hedging?
2. contextual_completeness: What share of the environmental boundary conditions stated in the text (temperature, flow, habitat, chemistry) are captured in the edge contexts, and are they correct?

Be strict. Use the full range. Deduct for every concrete error and do not default to high scores.

Return ONLY valid JSON:
{
  "causal_rigor_score": <float 0.0-1.0>,
  "contextual_completeness": <float 0.0-1.0>,
  "judge_rationale": "At most 120 words. Name concrete errors by edge_id, or state that none were found."
}
"""

SIM_SYSTEM_PROMPT = """You are the EcoAIRE counterfactual intervention reasoner.
Given extracted causal edges (with their environmental context moderators), a selected ecological intervention, and a DETERMINISTIC context-gate check, evaluate how the scientific hypothesis generalizes or breaks down under the new conditions.

The context gates were computed in code by comparing the intervention with the numeric thresholds stored in each edge:
- Satisfied: the intervention lies inside the condition under which the claim was reported.
- Violated: the intervention lies outside it, so the claim is not expected to hold.
- Partial: the ranges overlap.
- Not testable: no comparable numeric threshold exists.
Treat these gates as ground truth. Do not contradict a gate unless you explain why in the edge explanation.

Return ONLY valid JSON with this structure:
{
  "hypothesis_status": "Holds Strongly" | "Partially Attenuated" | "Reverses / Collapses Under Intervention",
  "predicted_invasion_risk": "High Risk" | "Moderate Risk" | "Low Risk (Biotic Resistance Restored)",
  "causal_chain_reasoning": "2-3 sentences explaining how the intervention propagates through the extracted causal edges and context moderators.",
  "edge_by_edge_impact": [
    {
      "edge_id": "E1",
      "pathway": "Cause -> Effect",
      "status_under_intervention": "Active / Amplified" | "Weakened" | "Blocked by Context Threshold",
      "explanation": "1 concise sentence citing the specific threshold from the paper."
    }
  ],
  "management_recommendation": "1 actionable river restoration recommendation grounded in the paper's evidence."
}
Return exactly one entry in edge_by_edge_impact for every extracted edge."""


# ----------------------------------------------------------------------------------------
# Extraction + evaluation
# ----------------------------------------------------------------------------------------
def run_extraction(
    client: Any,
    paper_id: str,
    paper_title: str,
    abstract_text: str,
    model_name: str = DEFAULT_MODEL,
    judge_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    judge_model_name = judge_model_name or JUDGE_MODEL

    user_prompt = (
        "Analyze the following scientific text and return the structured JSON representation:\n\n"
        f"PAPER ID: {paper_id}\nTITLE: {paper_title}\nTEXT / ABSTRACT:\n{abstract_text}\n"
    )

    def _validate_core(data: Dict[str, Any]) -> Dict[str, Any]:
        data = {**normalize_extraction(data), "paper_id": paper_id, "paper_title": paper_title}  # never trust echoed ids
        return ExtractionCore.model_validate(data).model_dump()

    core = chat_json(client, model_name, EXTRACTION_SYSTEM_PROMPT, user_prompt, 0.1, _validate_core)

    edges, stats = verify_edges(core["causal_edges"], abstract_text)
    core["causal_edges"] = edges
    diagnostics = analyze_causal_graph(edges)

    judge_user = json.dumps(
        {
            "source_text": abstract_text,
            "extraction": {
                "primary_hypothesis_name": core["primary_hypothesis_name"],
                "hypothesis_summary": core["hypothesis_summary"],
                "causal_edges": edges,
            },
            "programmatic_provenance_check": stats,
        },
        indent=2,
    )
    rigor: Optional[float] = None
    completeness: Optional[float] = None
    rationale = ""
    try:
        judged = chat_json(
            client, judge_model_name, JUDGE_SYSTEM_PROMPT, judge_user, 0.0,
            lambda d: JudgeOutput.model_validate(d).model_dump(),
        )
        rigor, completeness, rationale = (
            judged["causal_rigor_score"], judged["contextual_completeness"], judged["judge_rationale"],
        )
    except Exception as exc:  # the extraction is still useful without the judge
        rationale = f"LLM judge unavailable: {str(exc)[:200]}"

    evaluation = EvaluationMetrics(
        provenance_faithfulness=stats["faithfulness"],
        causal_rigor_score=rigor,
        contextual_completeness=completeness,
        judge_rationale=rationale,
        provenance_edges_exact=stats["exact"],
        provenance_edges_snapped=stats["snapped"],
        provenance_edges_unverified=stats["unverified"],
        provenance_edges_total=stats["total"],
    )
    result = HypothesisExtractionResult(
        **core,
        evaluation=evaluation,
        graph_diagnostics=diagnostics,
        model_name=model_name,
        judge_model=judge_model_name,
        source="live",
    )
    return result.model_dump()


# ----------------------------------------------------------------------------------------
# Intervention simulation
# ----------------------------------------------------------------------------------------
def _rows_with_gates(
    extraction_data: Dict[str, Any],
    llm_rows: List[Dict[str, Any]],
    gates: Dict[str, Dict[str, Any]],
    missing_status: str,
) -> List[Dict[str, Any]]:
    by_id = {r.get("edge_id"): r for r in llm_rows}
    rows = []
    for edge in extraction_data.get("causal_edges", []):
        eid = edge.get("edge_id", "")
        row = dict(by_id.get(eid) or {
            "edge_id": eid,
            "pathway": f"{edge.get('cause_variable')} -> {edge.get('effect_variable')}",
            "status_under_intervention": missing_status,
            "explanation": "",
        })
        gate = gates.get(eid, {"summary": "Not testable", "text": "Not testable"})
        row["context_gate"] = gate["text"]
        row["consistency"] = consistency_flag(row.get("status_under_intervention", ""), gate["summary"])
        rows.append(row)
    return rows


def run_simulation(
    client: Any,
    extraction_data: Dict[str, Any],
    temp_shift: str,
    flow_regime: str,
    bank_restoration: str,
    water_quality_regime: str,
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    params = {
        "temp_shift": temp_shift,
        "flow_regime": flow_regime,
        "bank_restoration": bank_restoration,
        "water_quality_regime": water_quality_regime,
    }
    gates = evaluate_context_gates(extraction_data.get("causal_edges", []), params)

    user_prompt = (
        "EXTRACTED CAUSAL GRAPH & CONTEXTS:\n"
        f"{json.dumps({k: extraction_data.get(k) for k in ('primary_hypothesis_name', 'hypothesis_summary', 'causal_edges')}, indent=2)}\n\n"
        "SELECTED INTERVENTION:\n"
        f"- Thermal Regime: {temp_shift}\n- Hydrological Flow: {flow_regime}\n"
        f"- Physical Habitat / Bank: {bank_restoration}\n- Water Quality / Nutrients / Salinity: {water_quality_regime}\n\n"
        "DETERMINISTIC CONTEXT GATES (ground truth):\n"
        f"{json.dumps({eid: g['text'] for eid, g in gates.items()}, indent=2)}\n\n"
        "Evaluate how the intervention alters the causal pathways and whether the primary hypothesis still holds."
    )
    sim = chat_json(
        client, model_name, SIM_SYSTEM_PROMPT, user_prompt, 0.15,
        lambda d: SimulationResult.model_validate(d).model_dump(),
    )
    sim["edge_by_edge_impact"] = _rows_with_gates(
        extraction_data, sim["edge_by_edge_impact"], gates, "Not returned by model"
    )
    conflicts = sum(1 for r in sim["edge_by_edge_impact"] if r["consistency"] == "Conflicts with context gate")
    sim["gate_conflicts"] = conflicts
    return sim


def deterministic_only_simulation(
    extraction_data: Dict[str, Any],
    temp_shift: str,
    flow_regime: str,
    bank_restoration: str,
    water_quality_regime: str,
) -> Dict[str, Any]:
    """Fallback used when the LLM is unavailable: only the code-computed gates are shown."""
    params = {
        "temp_shift": temp_shift,
        "flow_regime": flow_regime,
        "bank_restoration": bank_restoration,
        "water_quality_regime": water_quality_regime,
    }
    gates = evaluate_context_gates(extraction_data.get("causal_edges", []), params)
    rows = _rows_with_gates(extraction_data, [], gates, "n/a (LLM unavailable)")
    return {
        "hypothesis_status": "N/A (LLM unavailable)",
        "predicted_invasion_risk": "N/A (LLM unavailable)",
        "causal_chain_reasoning": (
            "The language model could not be reached. Only the deterministic context gates "
            "(computed from the extracted thresholds) are shown below."
        ),
        "edge_by_edge_impact": rows,
        "management_recommendation": "N/A (LLM unavailable)",
        "gate_conflicts": 0,
    }
