"""
algorithms/rl_feedback.py

Reinforcement Learning advice action selection via Q-Learning.
Manages a Q-table of state-action values persisted as a JSON file.

States:
  (risk_int, confidence_band)
  - risk_int: 0=low, 1=medium, 2=high
  - confidence_band: 0 if confidence < 0.7 else 1

Actions:
  - 0: mild_advice
  - 1: moderate_advice
  - 2: urgent_advice
  - 3: emergency_advice
"""

from __future__ import annotations
import json
import os
from pathlib import Path

_Q_TABLE_PATH = Path(__file__).parent.parent / "models" / "qtable.json"

# State mapping helper
_RISK_MAP = {"low": 0, "medium": 1, "high": 2}


def _load_q_table() -> list[list[list[float]]]:
    """Load Q-table from models/qtable.json. Initializes a 3x2x4 table of zeros if missing."""
    if _Q_TABLE_PATH.exists():
        try:
            with open(_Q_TABLE_PATH, "r") as f:
                q_table = json.load(f)
                # Verify dimensions (3, 2, 4)
                if len(q_table) == 3 and all(len(row) == 2 and len(col) == 4 for row in q_table for col in row):
                    return q_table
        except Exception:
            pass

    # Create directory if needed and save a default zero-initialized table
    _Q_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    default_table = [[[0.0 for _ in range(4)] for _ in range(2)] for _ in range(3)]
    
    # Pre-seed Q-table with logical defaults to guide the initial advice actions:
    # Low risk -> mild advice (action 0)
    default_table[0][0][0] = 0.5
    default_table[0][1][0] = 1.0
    # Medium risk -> moderate/urgent advice
    default_table[1][0][1] = 0.5
    default_table[1][1][1] = 1.0
    # High risk -> urgent/emergency advice
    default_table[2][0][2] = 0.5
    default_table[2][1][3] = 1.0
    
    _save_q_table(default_table)
    return default_table


def _save_q_table(q_table: list[list[list[float]]]) -> None:
    """Save the Q-table back to models/qtable.json."""
    with open(_Q_TABLE_PATH, "w") as f:
        json.dump(q_table, f, indent=2)


def _get_state_indices(risk_level: str, confidence: float) -> tuple[int, int]:
    """Map risk level string and confidence probability to Q-table indices."""
    risk_cleaned = risk_level.strip().lower()
    risk_int = _RISK_MAP.get(risk_cleaned, 0)
    confidence_band = 0 if confidence < 0.7 else 1
    return risk_int, confidence_band


# ── Public API ────────────────────────────────────────────────────────────────

def get_advice_action(risk_level: str, confidence: float) -> int:
    """
    Get the optimal action for the current state from the Q-table using argmax.

    Args:
        risk_level: 'low' | 'medium' | 'high'
        confidence: float confidence score (0.0-1.0)

    Returns:
        int action index (0=mild, 1=moderate, 2=urgent, 3=emergency)
    """
    q_table = _load_q_table()
    r_idx, c_idx = _get_state_indices(risk_level, confidence)
    
    q_values = q_table[r_idx][c_idx]
    
    # Return index of maximum value (argmax)
    # Simple Python implementation of argmax
    max_val = max(q_values)
    best_actions = [i for i, v in enumerate(q_values) if v == max_val]
    # Return first/lowest action index if there is a tie
    return best_actions[0]


def update_qtable(risk_level: str, confidence: float, action: int, reward: float) -> None:
    """
    Update the Q-table using the standard Q-learning update rule:
      Q(s, a) = Q(s, a) + alpha * [reward + gamma * max_a' Q(s', a') - Q(s, a)]
    Since we do not model state transition dynamics here (single-step MDP feedback),
    the future state s' is assumed to have the same state value distribution.

    Args:
        risk_level: 'low' | 'medium' | 'high'
        confidence: float confidence score (0.0-1.0)
        action: action index (0-3) that was chosen
        reward: scalar reward value (float) representing feedback quality
    """
    q_table = _load_q_table()
    r_idx, c_idx = _get_state_indices(risk_level, confidence)
    
    alpha = 0.1
    gamma = 0.9
    
    current_q = q_table[r_idx][c_idx][action]
    max_next_q = max(q_table[r_idx][c_idx]) # estimate next state's value as max of same state
    
    new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
    q_table[r_idx][c_idx][action] = round(new_q, 4)
    
    _save_q_table(q_table)
