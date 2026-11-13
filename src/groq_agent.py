import os
from typing import Any, Dict

import streamlit as st
from groq import Groq

from src import pipeline
from src.pipeline import DEFAULT_MODEL, EXTRACTION_SYSTEM_PROMPT, clean_and_parse_json  # noqa: F401

DEFAULT_GROQ_MODEL = DEFAULT_MODEL


def get_groq_client() -> Groq:
    """
    Retrieves the backend Groq API key from Streamlit secrets or environment variables
    so visiting reviewers never need to enter an API key.
    """
    api_key = None
    try:
        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "Backend GROQ_API_KEY not found. Add GROQ_API_KEY to .streamlit/secrets.toml, "
            "to Streamlit Cloud Secrets, or to the environment."
        )
    return Groq(api_key=api_key)


@st.cache_data(show_spinner=False)
def extract_causal_knowledge(
    paper_id: str,
    paper_title: str,
    abstract_text: str,
    model_name: str = DEFAULT_GROQ_MODEL,
) -> Dict[str, Any]:
    """Extraction, programmatic provenance check, graph diagnostics and a separate LLM-judge call."""
    return pipeline.run_extraction(get_groq_client(), paper_id, paper_title, abstract_text, model_name)


def simulate_ecological_intervention(
    extraction_data: Dict[str, Any],
    temp_shift: str,
    flow_regime: str,
    bank_restoration: str,
    water_quality_regime: str,
    model_name: str = DEFAULT_GROQ_MODEL,
) -> Dict[str, Any]:
    """LLM intervention reasoning that is constrained by deterministic context gates."""
    return pipeline.run_simulation(
        get_groq_client(), extraction_data, temp_shift, flow_regime,
        bank_restoration, water_quality_regime, model_name,
    )
