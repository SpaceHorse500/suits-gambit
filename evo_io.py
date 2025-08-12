# evo_io.py
from __future__ import annotations
import json
import copy
from typing import Any, Dict, Optional


# ---- Default genome (same structure/values I proposed) ----
DEFAULT_GENOME: Dict[str, Any] = {
    "forbidden": {
        "w_stats": 1.0,
        "w_match": 0.0,
        "w_random": 0.0,
        "w_anti": 0.0,
        "prefer_info_if_tied": 0.5,
        "gate_last_seat_boost_stats": 0.5,
        "gate_trailing_boost_stats": 0.5,
        "trail_threshold": 3.0
    },
    "draw_policy": {
        "min_target_plus": 3,
        "min_target_times": 3,
        "min_target_last_seat": 0,
        "min_target_r5_bump": 0,
        "plus_scale_base": 0.95,
        "times_scale_base": 0.85,
        "w_scale_lead": 0.04,
        "w_scale_trail": -0.04,
        "w_scale_is_last": -0.03,
        "w_scale_round_weight": 0.01,
        "w_scale_suit_skew": 0.03,
        "jitter": 0.02,
        "push_times_to3_if_p_bust_below": 0.33,
        "push_times_to4_if_p_bust_below": 0.15,
        "push_plus_to4_if_p_bust_below": 0.14,
        "brake_if_large_lead_r5": 6.0,
        "brake_scale_r5": 0.15,
        "bank5_bias_plus": 0.25,
        "bank5_bias_times": 0.15,
        "w_scale_overtake": 0.05
    },
    "ops_policy": {
        "x_prob_s_le2": 0.30,
        "x_prob_s_3_4": 0.45,
        "x_prob_s_ge5": 0.62,
        "w_xprob_trail": 0.10,
        "w_xprob_lead": -0.08,
        "w_xprob_is_last": 0.05,
        "w_xprob_round": -0.02,
        "x_repeat_s_le3": 0.10,
        "x_repeat_s_4_5": 0.18,
        "x_repeat_s_ge6": 0.30,
        "w_xrep_trail": 0.08,
        "w_xrep_lead": -0.06,
        "w_xrep_round": -0.02,
        "w_xprob_overtake": 0.08,
        "w_xrep_overtake": 0.05
    },
    "overtake_proxy": {
        "w_trail": 0.5,
        "w_rounds_left": 0.3,
        "w_is_last": 0.2
    },
    "fitness_penalty": {
        "seat_variance_weight": 0.05
    }
}


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """dst <- merge(src) without mutating src; src keys override dst."""
    out = copy.deepcopy(dst)
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)  # type: ignore
        else:
            out[k] = v
    return out


def load_genome(path: Optional[str]) -> Dict[str, Any]:
    """
    Load a genome JSON file and fill any missing fields with DEFAULT_GENOME.
    If path is None, return a deep copy of DEFAULT_GENOME.
    """
    if not path:
        return copy.deepcopy(DEFAULT_GENOME)
    with open(path, "r", encoding="utf-8") as f:
        user = json.load(f)
    return _deep_merge(DEFAULT_GENOME, user)


def save_genome(path: str, genome: Dict[str, Any]) -> None:
    """Save genome to JSON (pretty)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(genome, f, indent=2, ensure_ascii=False)
