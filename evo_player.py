# evo_player.py
from __future__ import annotations
import math
import random
from typing import Any, Dict, List, Optional

from players import BasePlayer
from cards import SUITS, Card
from utils import evaluate_expression
from evo_io import load_genome


# ---------- small math helpers ----------
def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))

def logit(p: float, eps: float = 1e-9) -> float:
    p = clamp(p, eps, 1 - eps)
    return math.log(p / (1 - p))

def softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    s = sum(exps)
    return [e / s for e in exps] if s > 0 else [1.0 / len(logits)] * len(logits)


# ---------- EvoPlayer ----------
class EvoPlayer(BasePlayer):
    """
    Context-aware, genome-driven bot.
    Reads a JSON genome (see evo_io.DEFAULT_GENOME) and makes decisions using:
      • table state (scores so far, ops so far)
      • round index and seat (first/last)
      • deck odds (p_bust)
    """

    def __init__(self, name: str, genome_path: Optional[str] = None, genome: Optional[Dict[str, Any]] = None):
        super().__init__(name)
        self.g = genome if genome is not None else load_genome(genome_path)
        self._forbidden: Optional[str] = None

    # ====== Helpers to read public context ======
    def _seat_info(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Derive seat position & flags from ctx['public']['turn_order']."""
        pub = ctx.get("public") or {}
        order: List[str] = pub.get("turn_order") or []
        try:
            seat_idx = order.index(self.name)  # 0-based
        except ValueError:
            seat_idx = 0
        n = len(order) if order else 1
        return {
            "seat_idx": seat_idx,
            "num_players": n,
            "is_first": seat_idx == 0,
            "is_last": seat_idx == (n - 1),
            "seat_pos": seat_idx + 1
        }

    def _table_totals(self, ctx: Dict[str, Any]) -> Dict[str, int]:
        """Evaluate partial totals for everyone based on scores/ops so far (finished rounds only)."""
        pub = ctx.get("public") or {}
        entries = pub.get("players_public") or []
        totals: Dict[str, int] = {}
        for e in entries:
            nm = e.get("name")
            sc = e.get("scores") or []
            ops = e.get("ops") or []
            try:
                totals[nm] = evaluate_expression(sc, ops)
            except Exception:
                totals[nm] = sum(sc) if sc else 0
        return totals

    def _lead_margins(self, ctx: Dict[str, Any]) -> Dict[str, float]:
        """My lead/trail vs table."""
        totals = self._table_totals(ctx)
        my = totals.get(self.name, 0)
        others = [v for k, v in totals.items() if k != self.name]
        max_others = max(others) if others else 0
        med_others = sorted(others)[len(others)//2] if others else 0
        return {
            "my_total": my,
            "max_others": max_others,
            "median_others": med_others,
            "lead_margin": my - max_others,       # >0 if I'm leading
            "median_margin": my - med_others
        }

    def _overtake_risk(self, rounds_left: int, lead_margin: float, is_last: bool) -> float:
        """Simple proxy in [0,1]; parameters from genome.overtake_proxy."""
        w = self.g["overtake_proxy"]
        # normalize: lead/deficit per 10 pts; rounds_left scaled by /5
        trail_norm = clamp(-lead_margin / 10.0, 0.0, 1.0)
        rl_norm = clamp(rounds_left / 5.0, 0.0, 1.0)
        x = w["w_trail"] * trail_norm + w["w_rounds_left"] * rl_norm + w["w_is_last"] * (1.0 if is_last else 0.0)
        return clamp(sigmoid(3 * (x - 0.5)), 0.0, 1.0)  # sharpen a little

    def _suit_concentration(self, rem_by: Dict[str, int]) -> float:
        """How skewed suits are; 1.0 ≈ very skewed, 0 ≈ uniform-ish."""
        if not rem_by:
            return 0.0
        total = sum(rem_by.values())
        if total <= 0:
            return 0.0
        ideal = total / 4.0
        mx = max(rem_by.values())
        return clamp((mx - ideal) / max(1.0, ideal), 0.0, 1.0)

    # ====== Forbidden suit ======
    def choose_forbidden_suit(self, first_revealed: Card, ctx: Dict[str, Any]) -> str:
        f = self.g["forbidden"]
        seat = self._seat_info(ctx)
        r_idx = ctx.get("round_index", 1)
        rounds_left = max(0, 5 - int(r_idx))

        # base logits
        L_stats = f["w_stats"]
        L_match = f["w_match"]
        L_random = f["w_random"]
        L_anti = f["w_anti"]

        # gates
        margins = self._lead_margins(ctx)
        if seat["is_last"] and rounds_left <= 2:
            L_stats += f["gate_last_seat_boost_stats"]
        if margins["lead_margin"] < -float(f["trail_threshold"]):
            L_stats += f["gate_trailing_boost_stats"]

        logits = [L_stats, L_match, L_random, L_anti]
        probs = softmax(logits)  # order: stats, match, random, anti

        # Stats you need
        rem_by: Dict[str, int] = ctx.get("deck_remaining_by_suit") or {}
        # pick a mode
        u = random.random()
        mode = 0 if u < probs[0] else 1 if u < probs[0] + probs[1] else 2 if u < probs[0] + probs[1] + probs[2] else 3

        # Mode behaviors
        if mode == 0 and rem_by and all(s in rem_by for s in SUITS):
            # stats: choose among suit(s) with FEWEST remaining (MOST revealed)
            min_left = min(rem_by[s] for s in SUITS)
            fewest = [s for s in SUITS if rem_by[s] == min_left]
            if first_revealed.suit in fewest and random.random() < f["prefer_info_if_tied"]:
                choice = first_revealed.suit
            else:
                choice = random.choice(fewest)
        elif mode == 1:
            choice = first_revealed.suit
        elif mode == 2:
            choice = random.choice(SUITS)
        else:
            # anti-info: avoid the info suit if possible
            alts = [s for s in SUITS if s != first_revealed.suit] or list(SUITS)
            choice = random.choice(alts)

        self._forbidden = choice
        return choice

    # ====== Continue or stop ======
    def choose_continue_or_stop(self, current_points: int, ctx: Dict[str, Any]) -> str:
        dp = self.g["draw_policy"]
        seat = self._seat_info(ctx)
        r_idx = int(ctx.get("round_index", 1))
        rounds_left = max(0, 5 - r_idx)
        last_op = self.ops_between[-1] if self.ops_between else "+"
        rem_by: Dict[str, int] = ctx.get("deck_remaining_by_suit") or {}
        forb = self._forbidden

        # p_bust
        remaining = sum(rem_by.values()) if rem_by else 0
        forbidden_left = rem_by.get(forb, 0) if (rem_by and forb in rem_by) else 0
        p_bust = (forbidden_left / remaining) if remaining > 0 else 1.0

        # features
        margins = self._lead_margins(ctx)
        lead = float(margins["lead_margin"])
        is_last = bool(seat["is_last"])
        suit_skew = self._suit_concentration(rem_by)
        risk_overtake = self._overtake_risk(rounds_left, lead, is_last)

        # min target adjustments
        min_target = dp["min_target_times"] if last_op == "x" else dp["min_target_plus"]
        min_target += dp["min_target_last_seat"] if is_last else 0
        if r_idx == 5:
            min_target += dp["min_target_r5_bump"]
        if current_points < min_target:
            return "continue"

        # EV-ish threshold with context
        base = dp["times_scale_base"] if last_op == "x" else dp["plus_scale_base"]
        scale = base
        # Lead/trail normalization: per 10 pts swing
        if lead >= 0:
            scale += dp["w_scale_lead"] * (lead / 10.0)
        else:
            scale += dp["w_scale_trail"] * (-lead / 10.0)
        scale += dp["w_scale_is_last"] * (1.0 if is_last else 0.0)
        scale += dp["w_scale_round_weight"] * (r_idx - 3.0)
        scale += dp["w_scale_suit_skew"] * suit_skew
        scale += dp["w_scale_overtake"] * risk_overtake

        thresh = scale / (current_points + 1.0)

        # Push/Brake rules
        if last_op == "x":
            if current_points < 3 and p_bust <= dp["push_times_to3_if_p_bust_below"]:
                return "continue"
            if current_points < 4 and p_bust <= dp["push_times_to4_if_p_bust_below"]:
                return "continue"
        else:
            if current_points < 4 and p_bust <= dp["push_plus_to4_if_p_bust_below"]:
                return "continue"

        if r_idx == 5 and lead >= dp["brake_if_large_lead_r5"]:
            thresh += dp["brake_scale_r5"]

        # Jitter to avoid determinism
        jitter = random.uniform(-dp["jitter"], dp["jitter"])
        # Bank@5 bias when marginal
        if current_points == 5:
            bias = dp["bank5_bias_times"] if last_op == "x" else dp["bank5_bias_plus"]
            # If we are just above/below the line, flip biased coin to bank
            margin = p_bust - (thresh + jitter)
            if -0.02 <= margin <= 0.02:  # marginal zone
                return "stop" if random.random() < bias else "continue"

        return "stop" if p_bust >= (thresh + jitter) else "continue"

    # ====== Operator between rounds ======
    def choose_operator_between_rounds(
        self,
        my_scores: List[int],
        all_scores: Dict[str, List[int]],
        previous_picks: List[Dict[str, str]],
        ctx: Dict[str, Any],
    ) -> str:
        opg = self.g["ops_policy"]
        seat = self._seat_info(ctx)
        r_idx = int(ctx.get("round_index", 1))
        rounds_left = max(0, 5 - r_idx)
        margins = self._lead_margins(ctx)
        lead = float(margins["lead_margin"])
        is_last = bool(seat["is_last"])
        last_score = my_scores[-1] if my_scores else 0

        # Never multiply from zero
        if last_score == 0:
            return "+"

        # Repeat × or start new?
        prev_op = self.ops_between[-1] if self.ops_between else None

        # Build a base probability based on last round score
        def base_prob_for_score(s: int, a: float, b: float, c: float) -> float:
            if s >= 5:
                return c
            elif s >= 3:
                return b
            else:
                return a

        # context scalers (logit space recommended; here we clamp)
        lead_norm = clamp(lead / 10.0, -1.0, 1.0)
        round_shift = (r_idx - 3.0) / 2.0  # -1 @R1 .. +1 @R5
        overtake = self._overtake_risk(rounds_left, lead, is_last)

        if prev_op == "x":
            p = base_prob_for_score(
                last_score,
                opg["x_repeat_s_le3"],
                opg["x_repeat_s_4_5"],
                opg["x_repeat_s_ge6"],
            )
            z = logit(p)
            z += opg["w_xrep_trail"] * clamp(-lead_norm, 0.0, 1.0)    # trailing → higher
            z += opg["w_xrep_lead"] * clamp(+lead_norm, 0.0, 1.0)     # leading → lower
            z += opg["w_xrep_round"] * round_shift
            z += opg["w_xrep_overtake"] * overtake
            p_final = clamp(sigmoid(z), 0.0, 1.0)
            return "x" if random.random() < p_final else "+"

        else:
            p = base_prob_for_score(
                last_score,
                opg["x_prob_s_le2"],
                opg["x_prob_s_3_4"],
                opg["x_prob_s_ge5"],
            )
            z = logit(p)
            z += opg["w_xprob_trail"] * clamp(-lead_norm, 0.0, 1.0)
            z += opg["w_xprob_lead"] * clamp(+lead_norm, 0.0, 1.0)
            z += opg["w_xprob_is_last"] * (1.0 if is_last else 0.0)
            z += opg["w_xprob_round"] * round_shift
            z += opg["w_xprob_overtake"] * overtake
            p_final = clamp(sigmoid(z), 0.0, 1.0)
            return "x" if random.random() < p_final else "+"
