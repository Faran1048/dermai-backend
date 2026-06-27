"""
algorithms/alpha_beta.py

Custom Minimax with Alpha-Beta Pruning.
Function: differential_diagnosis(confidence_json: dict, intake: dict) -> list

Builds a depth-3 game tree to find the best differential diagnoses.
- Depth 0 (Root): MAX — Selects the overall top diagnoses.
- Depth 1 (Level 1): Top 5 classes by confidence (MAX — maximize severity evidence).
- Depth 2 (Level 2): Benign evidence signals from intake (MIN — minimize severity/risk).
- Depth 3 (Level 3): Combined scores (Leaf nodes).
- Alpha-Beta prunes branches/leaves below threshold 0.15.
"""

from __future__ import annotations
from knowledge.graph import KNOWLEDGE_GRAPH

def _get_risk_level(score: float) -> str:
    """Determine risk level string based on evidence score thresholds."""
    if score >= 0.7:
        return "High"
    elif score >= 0.35:
        return "Medium"
    else:
        return "Low"

def differential_diagnosis(confidence_json: dict[str, float], intake: dict) -> list[dict]:
    """
    Evaluate differential diagnoses using minimax with alpha-beta pruning.

    Args:
        confidence_json: {class: probability} for all 7 classes.
        intake: Patient intake dict containing age, skin_type, duration_weeks,
                bleeding, growing, color_change, painful, itching, etc.

    Returns:
        Top 3 diagnoses sorted by evidence score descending:
        [{'class': str, 'evidence_score': float, 'risk_level': str}, ...]
    """
    # 1. Identify top 5 classes by confidence
    sorted_classes = sorted(confidence_json.items(), key=lambda x: x[1], reverse=True)
    top_5_classes = [cls for cls, _ in sorted_classes[:5]]

    # 2. Extract active benign evidence signals from intake (values of 0 indicate lack of symptom)
    # MIN player wants to minimize risk, so we look for benign factors (absence of risk symptoms).
    # If the symptom is absent (0) or duration is short, it counts as a benign signal.
    benign_signals = []
    
    if intake.get("bleeding") == 0:
        benign_signals.append(("no_bleeding", 0.30))
    if intake.get("growing") == 0:
        benign_signals.append(("no_growing", 0.25))
    if intake.get("color_change") == 0:
        benign_signals.append(("no_color_change", 0.20))
    if intake.get("painful") == 0:
        benign_signals.append(("no_painful", 0.15))
    if intake.get("itching") == 0:
        benign_signals.append(("no_itching", 0.10))
        
    duration = intake.get("duration_weeks", 0)
    if duration <= 4:
        benign_signals.append(("short_duration", 0.15))
        
    if not benign_signals:
        benign_signals.append(("none", 0.0))

    results = []

    # Run MiniMax with alpha-beta pruning for each of the top 5 candidates.
    # We prune any class node whose minimax value drops below 0.15.
    for cls in top_5_classes:
        confidence = confidence_json.get(cls, 0.0)
        urgency = KNOWLEDGE_GRAPH.get(cls, {}).get("urgency", 1.0)
        
        # We start with alpha = -infinity, beta = infinity.
        # Since we evaluate each class independently, we run a MIN search for each class.
        # MIN searches over the benign signals to find the one that minimizes the final score.
        # Score = confidence * (1.0 + urgency / 10.0) - benign_reduction
        
        best_val = float("inf")
        alpha = -float("inf")
        beta = float("inf")
        
        for name, reduction in benign_signals:
            # Leaf node score calculation
            leaf_score = confidence * (1.0 + urgency / 10.0) - reduction
            leaf_score = max(0.0, leaf_score)
            
            # MIN player minimizes
            if leaf_score < best_val:
                best_val = leaf_score
            
            beta = min(beta, best_val)
            # Alpha pruning threshold: if the score is below 0.15, we prune.
            if beta < 0.15:
                break
                
        # If the minimax score is at least 0.15, we keep the diagnosis.
        # If it was pruned (value < 0.15), we don't include it in final results or keep it with score 0.
        final_score = round(max(0.0, best_val), 4)
        if final_score >= 0.15:
            results.append({
                "class": cls,
                "evidence_score": final_score,
                "risk_level": _get_risk_level(final_score)
            })

    # Sort results by score descending and return top 3
    results.sort(key=lambda x: x["evidence_score"], reverse=True)
    return results[:3]
