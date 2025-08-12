# ga_bounds.py
from typing import Any, Dict, List, Tuple

FLOAT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "forbidden.w_stats": (-3.0, 3.0),
    "forbidden.w_match": (-3.0, 3.0),
    "forbidden.w_random": (-3.0, 3.0),
    "forbidden.w_anti": (-3.0, 3.0),
    "forbidden.prefer_info_if_tied": (0.0, 1.0),
    "forbidden.gate_last_seat_boost_stats": (-1.0, 1.5),
    "forbidden.gate_trailing_boost_stats": (-1.0, 1.5),
    "forbidden.trail_threshold": (0.0, 12.0),

    "draw_policy.plus_scale_base": (0.5, 1.4),
    "draw_policy.times_scale_base": (0.4, 1.4),
    "draw_policy.w_scale_lead": (-0.2, 0.2),
    "draw_policy.w_scale_trail": (-0.2, 0.2),
    "draw_policy.w_scale_is_last": (-0.2, 0.2),
    "draw_policy.w_scale_round_weight": (-0.1, 0.1),
    "draw_policy.w_scale_suit_skew": (-0.2, 0.2),
    "draw_policy.w_scale_overtake": (-0.2, 0.2),
    "draw_policy.jitter": (0.0, 0.06),
    "draw_policy.push_times_to3_if_p_bust_below": (0.05, 0.50),
    "draw_policy.push_times_to4_if_p_bust_below": (0.05, 0.40),
    "draw_policy.push_plus_to4_if_p_bust_below": (0.05, 0.40),
    "draw_policy.brake_if_large_lead_r5": (0.0, 20.0),
    "draw_policy.brake_scale_r5": (0.0, 0.5),
    "draw_policy.bank5_bias_plus": (0.0, 0.9),
    "draw_policy.bank5_bias_times": (0.0, 0.9),

    "ops_policy.x_prob_s_le2": (0.0, 0.9),
    "ops_policy.x_prob_s_3_4": (0.0, 0.9),
    "ops_policy.x_prob_s_ge5": (0.0, 0.95),
    "ops_policy.w_xprob_trail": (-0.3, 0.3),
    "ops_policy.w_xprob_lead": (-0.3, 0.3),
    "ops_policy.w_xprob_is_last": (-0.3, 0.3),
    "ops_policy.w_xprob_round": (-0.3, 0.3),
    "ops_policy.x_repeat_s_le3": (0.0, 0.9),
    "ops_policy.x_repeat_s_4_5": (0.0, 0.9),
    "ops_policy.x_repeat_s_ge6": (0.0, 0.95),
    "ops_policy.w_xrep_trail": (-0.3, 0.3),
    "ops_policy.w_xrep_lead": (-0.3, 0.3),
    "ops_policy.w_xrep_round": (-0.3, 0.3),
    "ops_policy.w_xprob_overtake": (-0.3, 0.3),
    "ops_policy.w_xrep_overtake": (-0.3, 0.3),

    "overtake_proxy.w_trail": (0.0, 1.5),
    "overtake_proxy.w_rounds_left": (0.0, 1.5),
    "overtake_proxy.w_is_last": (0.0, 1.5),

    "fitness_penalty.seat_variance_weight": (0.0, 0.5),
}

INT_KEYS: Dict[str, Tuple[int, int]] = {
    "draw_policy.min_target_plus": (2, 6),
    "draw_policy.min_target_times": (2, 6),
    "draw_policy.min_target_last_seat": (-1, 2),
    "draw_policy.min_target_r5_bump": (-1, 2),
}

def iter_paths(d: Dict[str, Any], prefix: str = "") -> List[str]:
    out: List[str] = []
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out += iter_paths(v, p)
        else:
            out.append(p)
    return out

def get_holder(d: Dict[str, Any], path: str):
    cur = d
    for k in path.split(".")[:-1]:
        cur = cur[k]
    return cur, path.split(".")[-1]

def clamp_to_bounds(path: str, val):
    if path in INT_KEYS:
        lo, hi = INT_KEYS[path]
        return int(max(lo, min(hi, round(val))))
    if path in FLOAT_BOUNDS:
        lo, hi = FLOAT_BOUNDS[path]
        return max(lo, min(hi, float(val)))
    return val

def range_for(path: str) -> Tuple[float, float]:
    if path in INT_KEYS:
        lo, hi = INT_KEYS[path]
        return float(lo), float(hi)
    if path in FLOAT_BOUNDS:
        return FLOAT_BOUNDS[path]
    return -1.0, 1.0
