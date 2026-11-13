from typing import Dict, Any, List
import networkx as nx
from pyvis.network import Network


def build_causal_networkx_graph(causal_edges: List[Dict[str, Any]]) -> nx.DiGraph:
    """
    Constructs a NetworkX Directed Graph from extracted causal edges.
    Normalizes node labels and attaches context + provenance metadata to edges.
    """
    G = nx.DiGraph()

    for edge in causal_edges:
        cause = edge.get("cause_variable", "Unknown Cause").strip()
        effect = edge.get("effect_variable", "Unknown Effect").strip()
        direction = edge.get("causal_direction", "Positive (+)")
        edge_id = edge.get("edge_id", "E")
        uncertainty = edge.get("uncertainty_level", "Moderate")
        mechanism = edge.get("mechanism_type", "Causal / Experimental")
        ctx = edge.get("context", {})
        quote = edge.get("provenance_quote", "")

        if not G.has_node(cause):
            G.add_node(cause, role="driver")
        if not G.has_node(effect):
            G.add_node(effect, role="outcome")

        G.add_edge(
            cause,
            effect,
            edge_id=edge_id,
            direction=direction,
            uncertainty=uncertainty,
            mechanism=mechanism,
            context=ctx,
            quote=quote,
        )

    return G


def render_interactive_pyvis_html(causal_edges: List[Dict[str, Any]], height_px: int = 440) -> str:
    """
    Converts extracted causal links into a self-contained interactive PyVis HTML string
    that embeds directly inside the single-screen Streamlit app.
    """
    G = build_causal_networkx_graph(causal_edges)

    net = Network(
        height=f"{height_px}px",
        width="100%",
        directed=True,
        bgcolor="#F8FAFC",
        font_color="#0F172A",
        notebook=False,
    )

    # Calculate node degrees to distinguish root drivers, mediators, and terminal outcomes
    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)

        if in_deg == 0 and out_deg > 0:
            node_color = "#0284C7"  # Blue: Root Driver / Intervention Variable
            group_label = "Driver / Cause Variable"
        elif out_deg == 0 and in_deg > 0:
            node_color = "#7C3AED"  # Purple: Ecological Outcome Variable
            group_label = "Ecological Outcome Variable"
        else:
            node_color = "#0D9488"  # Teal: Mediating Mechanism Variable
            group_label = "Mediating Ecological Process"

        net.add_node(
            node,
            label=node,
            title=f"<b>Variable:</b> {node}<br><b>Graph Role:</b> {group_label}",
            color={
                "background": node_color,
                "border": "#0F172A",
                "highlight": {"background": "#F59E0B", "border": "#0F172A"},
            },
            shape="box",
            margin=12,
            font={"color": "#FFFFFF", "size": 14, "face": "arial", "bold": True},
            borderWidth=2,
            shadow=True,
        )

    for u, v, data in G.edges(data=True):
        direction = data.get("direction", "Positive (+)")
        edge_id = data.get("edge_id", "E")
        uncertainty = data.get("uncertainty_level", data.get("uncertainty", "Moderate"))
        mechanism = data.get("mechanism", "Causal / Experimental")
        ctx = data.get("context", {})
        quote = data.get("quote", "")

        if "Positive" in direction or "+" in direction:
            edge_color = "#16A34A"  # Green for positive causal driver
            sign_symbol = "+"
        elif "Negative" in direction or "-" in direction:
            edge_color = "#DC2626"  # Red for negative / suppressive causal impact
            sign_symbol = "−"
        else:
            edge_color = "#64748B"  # Slate for null effect
            sign_symbol = "0"

        ctx_summary = (
            f"Temp: {ctx.get('water_temperature', 'Unspecified')} | "
            f"Flow: {ctx.get('flow_velocity', 'Unspecified')} | "
            f"Habitat: {ctx.get('habitat_structure', 'Unspecified')}"
        )

        hover_html = (
            f"<b>[{edge_id}] {u} → {v} ({direction})</b><br>"
            f"<b>Mechanism:</b> {mechanism} (Uncertainty: {uncertainty})<br>"
            f"<b>Boundary Context:</b> {ctx_summary}<br>"
            f"<b>Provenance Quote:</b> <i>\"{quote}\"</i>"
        )

        net.add_edge(
            u,
            v,
            label=f"{edge_id} ({sign_symbol})",
            title=hover_html,
            color=edge_color,
            width=3,
            arrows="to",
            dashes=(uncertainty == "High"),
            font={"align": "top", "size": 12, "color": "#1E293B", "strokeWidth": 3, "strokeColor": "#FFFFFF"},
        )

    net.set_options(
        """
        {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -4200,
              "centralGravity": 0.25,
              "springLength": 210,
              "springConstant": 0.04,
              "damping": 0.15,
              "avoidOverlap": 0.8
            },
            "stabilization": {
              "iterations": 150
            }
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "tooltipDelay": 100
          }
        }
        """
    )

    return net.generate_html()