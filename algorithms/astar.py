"""
algorithms/astar.py

Custom A* pathfinding for dermatology reasoning.

Builds a graph of symptom-state nodes and finds the shortest reasoning
path from the patient's observed symptoms to the predicted diagnosis node.

No external library — uses only Python's heapq and frozenset.
"""

from __future__ import annotations

import heapq
from typing import NamedTuple

# ── Graph definition ──────────────────────────────────────────────────────────
# Each node is a frozenset of symptom keywords that characterise a state.
# Edges connect symptom states that commonly co-occur or escalate.

NodeKey = frozenset  # type alias for readability

_SYMPTOM_NODES: dict[str, frozenset] = {
    "healthy":          frozenset({"no_symptoms"}),
    "mild_pigment":     frozenset({"light_brown", "small", "stable"}),
    "asymmetry":        frozenset({"asymmetric", "irregular_border"}),
    "colour_variation": frozenset({"multiple_colours", "dark_brown", "black"}),
    "regression":       frozenset({"blue_gray", "white_patches"}),
    "vascular":         frozenset({"red", "bleeding", "telangiectasia"}),
    "inflammatory":     frozenset({"red", "itching", "scaling"}),
    "melanoma_risk":    frozenset({"asymmetric", "multiple_colours", "growing", "dark_brown"}),
    "bcc_risk":         frozenset({"pearly", "telangiectasia", "bleeding", "nodule"}),
    "benign":           frozenset({"stable", "symmetric", "light_brown", "small"}),
}

# Adjacency: (neighbour_key, edge_weight)
_EDGES: dict[str, list[tuple[str, float]]] = {
    "healthy":          [("mild_pigment", 1.0), ("benign", 0.5)],
    "mild_pigment":     [("asymmetry", 1.5), ("benign", 0.5), ("colour_variation", 2.0)],
    "asymmetry":        [("colour_variation", 1.0), ("melanoma_risk", 1.5)],
    "colour_variation": [("regression", 1.5), ("melanoma_risk", 1.0)],
    "regression":       [("melanoma_risk", 0.5)],
    "vascular":         [("bcc_risk", 1.5), ("inflammatory", 1.0)],
    "inflammatory":     [("vascular", 1.0), ("asymmetry", 2.0)],
    "melanoma_risk":    [],
    "bcc_risk":         [],
    "benign":           [],
}

# Map predicted_class → goal node name
_CLASS_TO_GOAL: dict[str, str] = {
    "mel":    "melanoma_risk",
    "bcc":    "bcc_risk",
    "akiec":  "inflammatory",
    "nv":     "benign",
    "bkl":    "benign",
    "vasc":   "vascular",
    "df":     "benign",
}

# Human-readable labels for reasoning path
_NODE_LABELS: dict[str, str] = {
    "healthy":          "No significant findings",
    "mild_pigment":     "Mild pigmentation noted",
    "asymmetry":        "Detected asymmetry and irregular borders",
    "colour_variation": "High pigmentation asymmetry — multiple colour zones",
    "regression":       "Regression pattern (blue-grey) detected",
    "vascular":         "Vascular involvement — redness and bleeding",
    "inflammatory":     "Inflammatory response — itching and scaling",
    "melanoma_risk":    "Multiple colour zones → Melanoma risk",
    "bcc_risk":         "Pearly nodule with telangiectasia → BCC risk",
    "benign":           "Features consistent with benign lesion",
}


# ── Heuristic ─────────────────────────────────────────────────────────────────

def _heuristic(current: str, goal: str, symptoms: frozenset) -> float:
    """
    Admissible heuristic: fraction of goal-node symptoms NOT yet matched
    in the current node. Lower = closer to goal.
    """
    goal_symptoms = _SYMPTOM_NODES.get(goal, frozenset())
    current_symptoms = _SYMPTOM_NODES.get(current, frozenset())
    combined = current_symptoms | symptoms

    if not goal_symptoms:
        return 0.0

    matched = len(combined & goal_symptoms)
    return 1.0 - matched / len(goal_symptoms)


# ── A* ────────────────────────────────────────────────────────────────────────

def _astar(start: str, goal: str, symptoms: frozenset) -> list[str]:
    """
    Run A* from start to goal on the symptom-state graph.
    Returns the node path as a list of node-key strings.
    """
    # Priority queue entries: (f_score, node_key, path_so_far)
    open_heap: list[tuple[float, str, list[str]]] = []
    heapq.heappush(open_heap, (0.0, start, [start]))

    g_score: dict[str, float] = {start: 0.0}
    visited: set[str] = set()

    while open_heap:
        f, current, path = heapq.heappop(open_heap)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return path

        for neighbour, weight in _EDGES.get(current, []):
            if neighbour in visited:
                continue
            tentative_g = g_score.get(current, float("inf")) + weight
            if tentative_g < g_score.get(neighbour, float("inf")):
                g_score[neighbour] = tentative_g
                h = _heuristic(neighbour, goal, symptoms)
                f_new = tentative_g + h
                heapq.heappush(open_heap, (f_new, neighbour, path + [neighbour]))

    # If no path found, return direct route via goal
    return [start, goal]


# ── Public API ────────────────────────────────────────────────────────────────

def find_reasoning_path(symptoms: list[str], predicted_class: str) -> list[str]:
    """
    Compute the A* reasoning path from observed symptoms to the predicted
    diagnosis and return it as a list of human-readable step strings.

    Args:
        symptoms        : List of symptom/colour keywords from image analysis.
        predicted_class : ISIC class abbreviation (mel, nv, bcc, …).

    Returns:
        List of reasoning step strings, e.g.:
        [
            "Detected asymmetry and irregular borders",
            "High pigmentation asymmetry — multiple colour zones",
            "Multiple colour zones → Melanoma risk",
        ]
    """
    symptom_set = frozenset(s.lower().replace(" ", "_") for s in symptoms)
    goal = _CLASS_TO_GOAL.get(predicted_class, "benign")

    # Determine best start node by overlap with symptom_set
    best_start = "healthy"
    best_overlap = -1
    for node_name, node_symptoms in _SYMPTOM_NODES.items():
        overlap = len(symptom_set & node_symptoms)
        if overlap > best_overlap:
            best_overlap = overlap
            best_start = node_name

    if best_start == goal:
        # Already at goal
        return [_NODE_LABELS.get(goal, goal)]

    path_nodes = _astar(best_start, goal, symptom_set)

    return [_NODE_LABELS.get(node, node) for node in path_nodes]
