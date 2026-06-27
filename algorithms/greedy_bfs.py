"""
algorithms/greedy_bfs.py

Greedy Best-First Search style priority ranker for treatments.
Function: rank_treatments(lesion_class: str, symptoms: dict) -> list
"""

from __future__ import annotations
import heapq
from knowledge.treatment_kb import TREATMENT_KB

def rank_treatments(lesion_class: str, symptoms: dict) -> list[dict]:
    """
    Rank treatments for a given lesion class using priority scoring.
    Boosts priority scores if symptoms like bleeding (+0.2) or growing (+0.15) are present.

    Args:
        lesion_class: One of the 7 ISIC classes (e.g., 'mel', 'nv', etc.).
        symptoms: Dict of symptoms, e.g. {"bleeding": 1, "growing": 1}.

    Returns:
        Ordered list of top 3 treatment dicts.
    """
    treatments = TREATMENT_KB.get(lesion_class, [])
    if not treatments:
        return []

    heap: list[tuple[float, int, dict]] = []

    for idx, t_entry in enumerate(treatments):
        base_score = float(t_entry.get("priority_score", 0.0))
        boost = 0.0

        # Boost score if matching symptom flags
        if symptoms.get("bleeding") in (1, True):
            boost += 0.2
        if symptoms.get("growing") in (1, True):
            boost += 0.15

        boosted_score = base_score + boost
        # Priority for min-heap is negative of boosted score so that highest scores are smallest
        priority = -boosted_score

        # We construct a copy of the treatment dict and append the actual computed score for clarity
        result_dict = dict(t_entry)
        result_dict["boosted_priority_score"] = round(boosted_score, 4)

        heapq.heappush(heap, (priority, idx, result_dict))

    # Pop top 3 using heapq.nsmallest (which gets the most negative priority values = highest scores)
    top_3_entries = heapq.nsmallest(3, heap)

    # Return only the treatment dictionaries, ordered from highest score to lowest
    return [entry[2] for entry in top_3_entries]
