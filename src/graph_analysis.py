from collections import Counter
from itertools import islice
from typing import Any, Dict, List

import networkx as nx

from src.graph_builder import build_causal_networkx_graph

_SIGN = {"Positive (+)": 1, "Negative (-)": -1, "No Significant Effect (0)": 0}
_UNC_RANK = {"Low": 0, "Moderate": 1, "High": 2}
_UNC_NAME = {v: k for k, v in _UNC_RANK.items()}
MAX_PATHS = 25


def _sign_symbol(value: int) -> str:
    return {1: "+", -1: "-", 0: "0"}[value]


def analyze_causal_graph(edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    G = build_causal_networkx_graph(edges)

    pair_counts = Counter(
        (e.get("cause_variable", "").strip(), e.get("effect_variable", "").strip()) for e in edges
    )
    duplicate_pairs = [f"{c} -> {t}" for (c, t), n in pair_counts.items() if n > 1]

    is_dag = nx.is_directed_acyclic_graph(G)
    cycles = [c for c in islice(nx.simple_cycles(G), 3)] if not is_dag else []

    roots = sorted(n for n in G if G.in_degree(n) == 0 and G.out_degree(n) > 0)
    outcomes = sorted(n for n in G if G.out_degree(n) == 0 and G.in_degree(n) > 0)
    mediators = sorted(n for n in G if G.in_degree(n) > 0 and G.out_degree(n) > 0)

    paths: List[Dict[str, Any]] = []
    for r in roots:
        for o in outcomes:
            for nodes in nx.all_simple_paths(G, r, o, cutoff=6):
                sign, unc, ids = 1, 0, []
                for u, v in zip(nodes, nodes[1:]):
                    data = G[u][v]
                    sign *= _SIGN.get(data.get("direction", "Positive (+)"), 1)
                    unc = max(unc, _UNC_RANK.get(data.get("uncertainty", "Moderate"), 1))
                    ids.append(data.get("edge_id", ""))
                paths.append({
                    "nodes": nodes,
                    "edge_ids": ids,
                    "net_effect": _sign_symbol(sign),
                    "max_uncertainty": _UNC_NAME[unc],
                })
                if len(paths) >= MAX_PATHS:
                    break
            if len(paths) >= MAX_PATHS:
                break

    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "is_dag": is_dag,
        "cycles": cycles,
        "duplicate_cause_effect_pairs": duplicate_pairs,
        "root_drivers": roots,
        "mediators": mediators,
        "outcomes": outcomes,
        "driver_to_outcome_paths": paths,
        "note": (
            "Net effect is the qualitative product of edge signs along a path. "
            "It is not a quantitative causal effect estimate."
        ),
    }
