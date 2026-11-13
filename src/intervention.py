import re
from typing import Any, Dict, List, Optional, Tuple

INF = float("inf")
Interval = Tuple[float, float]

# Intervention options. Labels are what the UI shows; values are the region each one covers.
TEMP_OPTIONS: Dict[str, Interval] = {
    "Cool Shaded Regime (< 15°C)": (-INF, 15.0),
    "Baseline Ambient (16–18°C)": (16.0, 18.0),
    "Moderate Warming (+2°C / > 19°C)": (19.0, INF),
    "Extreme Summer Heatwave (> 22°C)": (22.0, INF),
}
FLOW_OPTIONS: Dict[str, Interval] = {
    "Restore High Lotic Flow (> 0.6 m/s via Dam Removal)": (0.6, INF),
    "Moderate Natural Flow (0.3–0.5 m/s)": (0.3, 0.5),
    "Low-Flow / Impounded Lentic (< 0.2 m/s)": (0.0, 0.2),
}
BANK_OPTIONS: Dict[str, str] = {
    "Remove Riprap -> Natural Gravel/Cobble & Canopy": "riprap_removed",
    "Mixed Bank Structure": "mixed",
    "Artificial Riprap Boulders Retained": "riprap_retained",
}
WATER_OPTIONS: Dict[str, Dict[str, Interval]] = {
    "Low Nutrient / Low Conductivity (< 200 µS/cm)": {"conductivity": (0.0, 200.0)},
    "Moderate Agricultural Runoff": {},
    "High Phosphorus (> 85 µg/L) & High Salinity (> 650 µS/cm)": {
        "phosphorus": (85.0, INF),
        "conductivity": (650.0, INF),
    },
}

_NUM = r"(\d+(?:\.\d+)?)"
_UNITS = {
    "temperature": r"°\s*C",
    "flow": r"m\s*/\s*s",
    "conductivity": r"[µμu]S\s*/\s*cm",
    "phosphorus": r"[µμu]g\s*/\s*L",
}
_OPS = {
    "<": "<", "<=": "<", "≤": "<", "below": "<", "under": "<",
    ">": ">", ">=": ">", "≥": ">", "above": ">", "over": ">", "exceeding": ">", "exceeds": ">",
}


def parse_condition(text: Optional[str], unit_key: str) -> Optional[Interval]:
    """Parse numeric bounds such as '< 0.25 m/s', '> 18.5°C' or '0.3–0.5 m/s' into an interval."""
    if not text:
        return None
    unit = _UNITS[unit_key]
    lo, hi, found = -INF, INF, False

    for m in re.finditer(rf"{_NUM}\s*(?:–|-|—|to)\s*{_NUM}\s*{unit}", text, flags=re.IGNORECASE):
        lo, hi, found = max(lo, float(m.group(1))), min(hi, float(m.group(2))), True

    op_pattern = "|".join(sorted((re.escape(k) for k in _OPS), key=len, reverse=True))
    for m in re.finditer(rf"({op_pattern})\s*{_NUM}\s*{unit}", text, flags=re.IGNORECASE):
        op, value = _OPS[m.group(1).lower()], float(m.group(2))
        if op == ">":
            lo = max(lo, value)
        else:
            hi = min(hi, value)
        found = True

    if not found or lo > hi:
        return None
    return (lo, hi)


def relation(intervention: Interval, condition: Interval) -> str:
    """Satisfied: intervention lies inside the stated condition. Violated: disjoint. Else Partial."""
    if intervention[0] >= condition[0] and intervention[1] <= condition[1]:
        return "Satisfied"
    if intervention[1] <= condition[0] or intervention[0] >= condition[1]:
        return "Violated"
    return "Partial"


def _habitat_gate(habitat_text: Optional[str], bank_choice: Optional[str]) -> Optional[str]:
    """Keyword rule: does the edge require riprap, or a natural (gravel/cobble) bank?"""
    t = (habitat_text or "").lower()
    needs_riprap = "riprap" in t
    needs_natural = any(w in t for w in ("natural", "gravel", "cobble"))
    if bank_choice is None or needs_riprap == needs_natural:
        return None
    table = {
        True: {"riprap_retained": "Satisfied", "mixed": "Partial", "riprap_removed": "Violated"},
        False: {"riprap_retained": "Violated", "mixed": "Partial", "riprap_removed": "Satisfied"},
    }
    return table[needs_riprap].get(bank_choice)


def _summary(statuses: List[str]) -> str:
    if not statuses:
        return "Not testable"
    if "Violated" in statuses:
        return "Violated"
    if all(s == "Satisfied" for s in statuses):
        return "Satisfied"
    return "Partial"


def evaluate_context_gates(edges: List[Dict[str, Any]], params: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Return {edge_id: {dimensions, summary, text}} for the selected intervention."""
    regions: Dict[str, Optional[Interval]] = {
        "temperature": TEMP_OPTIONS.get(params.get("temp_shift", "")),
        "flow": FLOW_OPTIONS.get(params.get("flow_regime", "")),
    }
    regions.update(WATER_OPTIONS.get(params.get("water_quality_regime", ""), {}))
    bank_choice = BANK_OPTIONS.get(params.get("bank_restoration", ""))

    out: Dict[str, Dict[str, Any]] = {}
    for edge in edges:
        ctx = edge.get("context", {}) or {}
        sources = {
            "temperature": ctx.get("water_temperature"),
            "flow": ctx.get("flow_velocity"),
            "conductivity": ctx.get("additional_moderators"),
            "phosphorus": ctx.get("additional_moderators"),
        }
        dims: Dict[str, str] = {}
        for dim, text in sources.items():
            region = regions.get(dim)
            cond = parse_condition(text, dim)
            if region is not None and cond is not None:
                dims[dim] = relation(region, cond)
        habitat = _habitat_gate(ctx.get("habitat_structure"), bank_choice)
        if habitat:
            dims["habitat"] = habitat

        summary = _summary(list(dims.values()))
        text = " | ".join(f"{k}: {v}" for k, v in dims.items()) or "No testable thresholds"
        out[edge.get("edge_id", "")] = {"dimensions": dims, "summary": summary, "text": f"{summary} ({text})"}
    return out


def consistency_flag(status: str, gate_summary: str) -> str:
    """Compare the LLM's per-edge verdict with the deterministic gate."""
    s = (status or "").lower()
    active = ("active" in s or "amplified" in s) and "block" not in s and "weak" not in s
    if gate_summary == "Violated" and active:
        return "Conflicts with context gate"
    if gate_summary == "Satisfied" and "block" in s:
        return "Conflicts with context gate"
    if gate_summary == "Not testable":
        return "n/a"
    return "Consistent"
