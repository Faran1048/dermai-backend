"""
algorithms/dijkstra.py

Computes the shortest referral pathway using Dijkstra's algorithm.
Function: referral_pathway(lesion_class: str, risk_level: str) -> dict
"""

from __future__ import annotations
import networkx as nx
from knowledge.graph import KNOWLEDGE_GRAPH

def referral_pathway(lesion_class: str, risk_level: str) -> dict:
    """
    Build a NetworkX DiGraph of referral steps and find the shortest path from
    'self_monitor' to the recommended goal pathway.

    Args:
        lesion_class : ISIC abbreviation (e.g., 'mel', 'bcc', 'nv')
        risk_level   : 'Low', 'Medium', or 'High'

    Returns:
        {
            "path"               : list of strings (referral nodes)
            "estimated_timeline" : str
            "urgency"            : str ('Low' | 'Medium' | 'High')
        }
    """
    # 1. Initialize directed graph and nodes
    G = nx.DiGraph()
    nodes = ["self_monitor", "gp", "dermatologist", "biopsy", "oncology"]
    G.add_nodes_from(nodes)

    # 2. Get pathway goal and urgency score from knowledge graph
    class_info = KNOWLEDGE_GRAPH.get(lesion_class, {"pathway": "self_monitor", "urgency": 1})
    goal = class_info.get("pathway", "self_monitor")
    if goal not in nodes:
        goal = "self_monitor"

    urgency_val = class_info.get("urgency", 1)
    
    # 3. Determine actual clinical urgency class
    is_high = (risk_level.lower() == "high") or (urgency_val >= 7)
    is_med = (risk_level.lower() == "medium") or (4 <= urgency_val <= 6)

    # 4. Define transition weights based on clinical urgency.
    # Higher urgency/risk reduces weights on shortcuts to oncology/biopsy,
    # ensuring they are selected as shortest paths, while low urgency forces
    # longer routing or simple self-monitoring.
    if is_high:
        G.add_edge("self_monitor", "gp", weight=5.0)
        G.add_edge("self_monitor", "dermatologist", weight=3.0)
        G.add_edge("self_monitor", "oncology", weight=1.0)  # Direct fast-track
        G.add_edge("gp", "dermatologist", weight=1.0)
        G.add_edge("gp", "oncology", weight=1.0)
        G.add_edge("dermatologist", "biopsy", weight=1.0)
        G.add_edge("biopsy", "oncology", weight=0.5)
        timeline = "Urgent — Same-week referral / biopsy within 48 hours"
        urgency = "High"
    elif is_med:
        G.add_edge("self_monitor", "gp", weight=1.0)
        G.add_edge("self_monitor", "dermatologist", weight=2.0)
        G.add_edge("self_monitor", "oncology", weight=8.0)
        G.add_edge("gp", "dermatologist", weight=1.0)
        G.add_edge("dermatologist", "biopsy", weight=1.5)
        G.add_edge("biopsy", "oncology", weight=2.0)
        timeline = "Routine Specialist Visit — Schedule within 1-2 weeks"
        urgency = "Medium"
    else:
        G.add_edge("self_monitor", "gp", weight=1.0)
        G.add_edge("self_monitor", "dermatologist", weight=10.0)
        G.add_edge("self_monitor", "oncology", weight=20.0)
        G.add_edge("gp", "dermatologist", weight=5.0)
        G.add_edge("dermatologist", "biopsy", weight=5.0)
        G.add_edge("biopsy", "oncology", weight=10.0)
        timeline = "Observation — Regular self-monitoring or annual check-up"
        urgency = "Low"

    # 5. Compute shortest path using Dijkstra's algorithm
    try:
        path = nx.shortest_path(G, source="self_monitor", target=goal, weight="weight")
    except Exception:
        # Fallback in case of disconnected path (should not happen with these edges)
        path = ["self_monitor", goal]

    return {
        "path": path,
        "estimated_timeline": timeline,
        "urgency": urgency,
    }
