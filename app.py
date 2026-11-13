import json
import os
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

from src.groq_agent import extract_causal_knowledge, simulate_ecological_intervention
from src.graph_builder import render_interactive_pyvis_html
from src.cache import get_precomputed
from src.pipeline import deterministic_only_simulation
from src.intervention import TEMP_OPTIONS, FLOW_OPTIONS, BANK_OPTIONS, WATER_OPTIONS
from src.report_generator import generate_markdown_report, generate_orkg_jsonld_export

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EcoAIRE | Causal AI & Scientific Reasoning Prototype (TIB Hannover)",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom styling for clickable paper cards & academic badges
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #0F172A 0%, #005A9C 100%);
        color: white;
        padding: 1.4rem 1.8rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
    }
    .main-header h1 {
        color: white !important;
        font-size: 1.75rem !important;
        margin-bottom: 0.3rem !important;
    }
    .main-header p {
        color: #E2E8F0 !important;
        font-size: 0.96rem !important;
        margin: 0 !important;
    }
    .paper-card {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 10px;
        padding: 0.9rem;
        height: 215px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    .paper-card-active {
        background-color: #EFF6FF;
        border: 2px solid #005A9C;
        border-radius: 10px;
        padding: 0.9rem;
        height: 215px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        box-shadow: 0 4px 12px rgba(0, 90, 156, 0.12);
    }
    .badge {
        display: inline-block;
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.18rem 0.55rem;
        border-radius: 999px;
        font-size: 0.73rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# LOAD SAMPLE PAPERS
# -----------------------------------------------------------------------------
@st.cache_data
def load_sample_papers():
    data_path = Path(__file__).parent / "data" / "sample_papers.json"
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


papers = load_sample_papers()

SYNTHETIC_NOTICE = (
    "⚠️ All sample papers are **synthetic illustrative abstracts** written to demonstrate the pipeline; "
    "they are not real publications. Provenance is checked in code; the other scores come from an LLM judge "
    "and are not benchmark results."
)


def _pct(value):
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _prov_badge(edge):
    status = edge.get("provenance_status")
    return {
        "exact": "✅ *found verbatim in source*",
        "snapped": "🔧 *model quote replaced by closest source sentence*",
        "unverified": "⚠️ *not found in source*",
    }.get(status, "")

# Initialize session state
if "selected_paper" not in st.session_state:
    st.session_state.selected_paper = papers[0]
if "custom_mode" not in st.session_state:
    st.session_state.custom_mode = False
if "sim_result" not in st.session_state:
    st.session_state.sim_result = None
if "sim_params" not in st.session_state:
    st.session_state.sim_params = None

# -----------------------------------------------------------------------------
# HEADER BANNER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🌿 EcoAIRE-Lite: Provenance-Aware Causal AI & Scientific Hypothesis Simulator</h1>
        <p>
            Interactive Research Prototype for <b>TIB Hannover — Project EcoAIRE (Ref. No. 44/2026)</b> | 
            Developed by <b>Md Shahid Afridi</b> (M.Sc. Computer Science — Big Data Analytics)
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# SECTION 1: CLICKABLE 5 RESEARCH PAPERS + CUSTOM PAPER INPUT ON ONE SCREEN
# -----------------------------------------------------------------------------
st.caption(SYNTHETIC_NOTICE)
st.subheader("1️⃣ Select a Freshwater Invasion Biology Paper (or Test Your Own Research Paper)")
st.caption(
    "Click any of the 5 synthetic illustrative abstracts below to immediately run structured causal extraction, "
    "ORKG semantic triple alignment, and world-model intervention reasoning—or click **➕ Test Your Own Research Paper**."
)

cols = st.columns(5)
for idx, paper in enumerate(papers):
    with cols[idx]:
        is_active = (
            not st.session_state.custom_mode
            and st.session_state.selected_paper["id"] == paper["id"]
        )
        card_class = "paper-card-active" if is_active else "paper-card"
        st.markdown(
            f"""
            <div class="{card_class}">
                <div>
                    <span class="badge">{paper['hypothesis_badge']}</span>
                    <div style="font-weight:700; font-size:0.86rem; color:#0F172A; line-height:1.25; margin-top:4px;">
                        {paper['short_tag']}
                    </div>
                    <div style="font-size:0.75rem; color:#475569; margin-top:6px;">
                        {paper['authors']} ({paper['year']})
                    </div>
                </div>
                <div style="font-size:0.72rem; color:#005A9C; font-weight:600;">
                    Source: {paper['doi']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        btn_label = "✅ Active in Model" if is_active else f"▶ Run {paper['id']}"
        if st.button(
            btn_label,
            key=f"btn_{paper['id']}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.selected_paper = paper
            st.session_state.custom_mode = False
            st.session_state.sim_result = None
            st.rerun()

# Custom Paper Expander right underneath the 5 cards
with st.expander(
    "➕ Try Your Own Research Paper / Scientific Abstract (Live Custom Input)",
    expanded=st.session_state.custom_mode,
):
    with st.form("custom_paper_form"):
        custom_title = st.text_input(
            "Paper Title",
            placeholder="e.g., Restoring riparian shading suppresses invasive macrophytes in lowland streams",
        )
        custom_abstract = st.text_area(
            "Paste Abstract or Results Section (Include variables, environmental conditions, or experimental findings)",
            height=140,
            placeholder="Paste any ecology, invasion biology, or scientific abstract here...",
        )
        submitted_custom = st.form_submit_button(
            "🚀 Run EcoAIRE Causal Model on Custom Paper", type="primary"
        )
        if submitted_custom and custom_abstract.strip():
            st.session_state.selected_paper = {
                "id": "CUSTOM_PAPER",
                "short_tag": "Custom User Paper",
                "title": custom_title.strip() or "Custom User-Submitted Scientific Paper",
                "authors": "User / Hiring Committee Test",
                "year": 2026,
                "journal": "Live Interactive Test",
                "doi": "custom-input",
                "hypothesis_badge": "Custom Hypothesis Assessment",
                "abstract_text": custom_abstract.strip(),
            }
            st.session_state.custom_mode = True
            st.session_state.sim_result = None
            st.rerun()

active_paper = st.session_state.selected_paper

# Show the selected paper text
with st.container(border=True):
    st.markdown(f"#### 📄 Active Source Text: *{active_paper['title']}*")
    st.caption(
        f"**Authors:** {active_paper['authors']} ({active_paper['year']}) | "
        f"**Journal:** {active_paper['journal']} | **ID:** `{active_paper['id']}`"
    )
    st.write(active_paper["abstract_text"])

# -----------------------------------------------------------------------------
# SECTION 2: RUN MODEL EXTRACTION & DISPLAY CAUSAL GRAPH + PROVENANCE
# -----------------------------------------------------------------------------
try:
    if os.environ.get("ECOAIRE_OFFLINE") == "1":
        raise RuntimeError("offline mode (ECOAIRE_OFFLINE=1)")
    with st.spinner("🧠 Running EcoAIRE Agent (Structured Causal Extraction + ORKG Alignment + LLM-as-a-Judge)..."):
        extraction_result = extract_causal_knowledge(
            paper_id=active_paper["id"],
            paper_title=active_paper["title"],
            abstract_text=active_paper["abstract_text"],
        )
except Exception as e:
    cached = get_precomputed(active_paper["id"], active_paper["abstract_text"])
    if cached is None:
        st.error(f"Model Execution Error: {str(e)[:300]}")
        st.stop()
    extraction_result = cached["result"]
    st.warning(
        f"Live model unavailable ({str(e)[:120]}). Showing a precomputed result generated on "
        f"{cached['meta'].get('generated_utc', 'an unknown date')} with {cached['meta'].get('model', 'an unknown model')}."
    )

st.divider()
st.subheader("2️⃣ Extracted Causal Knowledge Graph & Provenance-Grounded Evidence")

# Top synthesis bar
m_col1, m_col2, m_col3, m_col4 = st.columns([2.2, 1, 1, 1])
with m_col1:
    st.info(
        f"**Primary Hypothesis:** `{extraction_result['primary_hypothesis_name']}`\n\n"
        f"{extraction_result['hypothesis_summary']}"
    )
eval_scores = extraction_result.get("evaluation", {})
with m_col2:
    st.metric(
        "Provenance Faithfulness",
        _pct(eval_scores.get('provenance_faithfulness')),
        "Verified in code",
    )
with m_col3:
    st.metric(
        "Causal vs. Correlational Rigor",
        _pct(eval_scores.get('causal_rigor_score')),
        "LLM-judged",
    )
with m_col4:
    st.metric(
        "Context Completeness",
        _pct(eval_scores.get('contextual_completeness')),
        "LLM-judged",
    )

# Side-by-side: Interactive Causal DAG on left, Structured Provenance & ORKG tabs on right
graph_col, table_col = st.columns([1.15, 1.25])

with graph_col:
    st.markdown("##### 🕸️ Interactive Causal Directed Acyclic Graph (DAG)")
    st.caption(
        "🔵 **Blue:** Root Driver | 🟢 **Teal:** Mediator | 🟣 **Purple:** Outcome | "
        "**Green Edge (+):** Positive Effect | **Red Edge (−):** Negative Effect *(Hover over nodes/edges for exact provenance)*"
    )
    html_graph = render_interactive_pyvis_html(extraction_result.get("causal_edges", []), height_px=430)
    components.html(html_graph, height=445, scrolling=False)

with table_col:
    tab_edges, tab_orkg, tab_judge = st.tabs([
        "🔍 Causal Claims & Context Moderators",
        "🌐 ORKG Semantic Triples (JSON-LD)",
        "⚖️ LLM-as-a-Judge Audit",
    ])

    with tab_edges:
        for edge in extraction_result.get("causal_edges", []):
            ctx = edge.get("context", {})
            with st.container(border=True):
                st.markdown(
                    f"**[{edge['edge_id']}] `{edge['cause_variable']}` ➡️ `{edge['effect_variable']}`** "
                    f"— **{edge['causal_direction']}** *({edge['mechanism_type']}, Uncertainty: {edge['uncertainty_level']})*"
                )
                st.caption(
                    f"🌡️ **Temp:** `{ctx.get('water_temperature', 'Unspecified')}` | "
                    f"🌊 **Flow:** `{ctx.get('flow_velocity', 'Unspecified')}` | "
                    f"🪨 **Habitat:** `{ctx.get('habitat_structure', 'Unspecified')}`"
                    + (f" | 🧪 **Other:** `{ctx.get('additional_moderators')}`" if ctx.get("additional_moderators") else "")
                )
                st.markdown(f"> 📌 **Verbatim Provenance:** *\"{edge['provenance_quote']}\"* {_prov_badge(edge)}")

    with tab_orkg:
        st.caption(
            "Machine-actionable Semantic Web triples formatted for the **Open Research Knowledge Graph (ORKG)** at TIB Hannover:"
        )
        st.dataframe(
            extraction_result.get("orkg_triples", []),
            use_container_width=True,
            hide_index=True,
        )

    with tab_judge:
        st.markdown("##### Automated Faithfulness & Error-Severity Critique")
        st.write(eval_scores.get("judge_rationale", ""))
        st.json(eval_scores)
        diag = extraction_result.get("graph_diagnostics") or {}
        if diag:
            st.markdown("##### Programmatic Graph Checks")
            st.caption(
                f"Acyclic (DAG): **{'yes' if diag.get('is_dag') else 'NO — cycle detected'}** | "
                f"Nodes: {diag.get('n_nodes')} | Edges: {diag.get('n_edges')} | "
                f"Provenance: {eval_scores.get('provenance_edges_exact', 0)}/{eval_scores.get('provenance_edges_total', 0)} exact"
            )
            if diag.get("driver_to_outcome_paths"):
                st.dataframe(
                    [
                        {
                            "path": " → ".join(p["nodes"]),
                            "edges": ", ".join(p["edge_ids"]),
                            "net effect": p["net_effect"],
                            "max uncertainty": p["max_uncertainty"],
                        }
                        for p in diag["driver_to_outcome_paths"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            st.caption(diag.get("note", ""))

# -----------------------------------------------------------------------------
# SECTION 3: WORLD-MODEL COUNTERFACTUAL INTERVENTION SIMULATOR
# -----------------------------------------------------------------------------
st.divider()
st.subheader("3️⃣ Causal World-Model & Counterfactual Intervention Simulator")
st.caption(
    "Test how the paper's ecological hypothesis generalizes or breaks down when river managers apply restoration "
    "interventions or when climate conditions shift. Context gates are computed deterministically from the extracted thresholds; "
    "status and narrative come from an LLM."
)

sim_col1, sim_col2, sim_col3, sim_col4, sim_col5 = st.columns([1, 1, 1, 1, 0.8])

with sim_col1:
    temp_shift = st.selectbox(
        "🌡️ Thermal Regime Shift",
        list(TEMP_OPTIONS),
        index=0,
    )
with sim_col2:
    flow_regime = st.selectbox(
        "🌊 Hydrological Flow Intervention",
        list(FLOW_OPTIONS),
        index=0,
    )
with sim_col3:
    bank_restoration = st.selectbox(
        "🪨 Habitat / Bank Intervention",
        list(BANK_OPTIONS),
        index=0,
    )
with sim_col4:
    water_quality_regime = st.selectbox(
        "🧪 Water Quality / Salinity / Nutrients",
        list(WATER_OPTIONS),
        index=0,
    )
with sim_col5:
    st.write("")
    st.write("")
    run_sim = st.button("⚡ Simulate Intervention", type="primary", use_container_width=True)

if run_sim:
    with st.spinner("🔬 Propagating counterfactual intervention across causal graph..."):
        st.session_state.sim_params = {
            "temp_shift": temp_shift,
            "flow_regime": flow_regime,
            "bank_restoration": bank_restoration,
            "water_quality_regime": water_quality_regime,
        }
        try:
            if os.environ.get("ECOAIRE_OFFLINE") == "1":
                raise RuntimeError("offline mode (ECOAIRE_OFFLINE=1)")
            st.session_state.sim_result = simulate_ecological_intervention(
                extraction_data=extraction_result,
                temp_shift=temp_shift,
                flow_regime=flow_regime,
                bank_restoration=bank_restoration,
                water_quality_regime=water_quality_regime,
            )
        except Exception as e:
            st.warning(f"LLM unavailable ({str(e)[:120]}). Showing deterministic context gates only.")
            st.session_state.sim_result = deterministic_only_simulation(
                extraction_data=extraction_result,
                temp_shift=temp_shift,
                flow_regime=flow_regime,
                bank_restoration=bank_restoration,
                water_quality_regime=water_quality_regime,
            )

if st.session_state.sim_result:
    sim = st.session_state.sim_result
    with st.container(border=True):
        s_col1, s_col2 = st.columns([1, 2])
        with s_col1:
            st.metric("Hypothesis Generalization Status", sim.get("hypothesis_status", "N/A"))
            st.metric("Predicted Post-Intervention Risk", sim.get("predicted_invasion_risk", "N/A"))
        with s_col2:
            st.markdown(f"**🧠 Causal Propagation Reasoning:**\n\n{sim.get('causal_chain_reasoning', '')}")
            st.success(f"**🛠️ Restoration Recommendation:** {sim.get('management_recommendation', '')}")

        if sim.get("gate_conflicts"):
            st.warning(
                f"{sim['gate_conflicts']} edge verdict(s) from the LLM conflict with the deterministic context gate "
                "(see the 'consistency' column)."
            )
        st.markdown("###### Edge-by-Edge Causal Pathway Impact Under Intervention")
        st.dataframe(sim.get("edge_by_edge_impact", []), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# SECTION 4: ONE-CLICK DOWNLOADABLE SCIENTIFIC ASSESSMENT REPORT
# -----------------------------------------------------------------------------
st.divider()
st.subheader("4️⃣ Download Complete Scientific Reasoning & ORKG Report")
st.caption(
    "Export the extracted causal graph, verbatim provenance quotes, ORKG semantic triples, counterfactual simulation results, "
    "and LLM-as-a-Judge evaluation scores."
)

md_report_str = generate_markdown_report(
    extraction_data=extraction_result,
    simulation_data=st.session_state.sim_result,
    sim_params=st.session_state.sim_params,
)
jsonld_report_str = generate_orkg_jsonld_export(
    extraction_data=extraction_result,
    simulation_data=st.session_state.sim_result,
)

dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 1.2])
with dl_col1:
    st.download_button(
        label="📥 Download Full Assessment Report (.md)",
        data=md_report_str,
        file_name=f"EcoAIRE_Causal_Report_{active_paper['id']}.md",
        mime="text/markdown",
        type="primary",
        use_container_width=True,
    )
with dl_col2:
    st.download_button(
        label="🌐 Download ORKG Semantic Graph (.jsonld)",
        data=jsonld_report_str,
        file_name=f"EcoAIRE_ORKG_Triples_{active_paper['id']}.json",
        mime="application/ld+json",
        use_container_width=True,
    )
with dl_col3:
    with st.popover("👁️ Preview Markdown Report Before Downloading", use_container_width=True):
        st.markdown(md_report_str)