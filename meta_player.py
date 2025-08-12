# meta_control_v1.py
import random
from typing import List, Dict, Any, Optional
from players import BasePlayer
from cards import SUITS, Card


class MetaPlayer(BasePlayer):
    """
    MetaPlayer — control bot tuned to your current meta (anchor-style like EvoGen019).

    Key ideas:
      • Forbidden suit: prefer STATS (fewest remaining) with smart gating; small anti-info spice.
      • Stop policy: push to 3, selectively to 4 based on p_bust and table context; rare @5.
      • Operator policy: ~50% × overall, but damp when leading / R5; boost when trailing/last seat.

    Behavior targets:
      • Bust rate ≈ 43–47% in mixed fields.
      • Stops: heavy @3, selective @4, rare @5.
      • Pairwise resilience vs additive-risky pool; doesn’t donate when leading late.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self._forbidden: Optional[str] = None

    # ----------------- tiny helpers -----------------
    @staticmethod
    def _eval_expr(scores: List[int], ops: List[str]) -> int:
        """Evaluate scores with ops (+ or x). Fallback to sum on mismatch."""
        if not scores:
            return 0
        if not ops or len(ops) != len(scores) - 1:
            return sum(scores)
        total = scores[0]
        for i in range(1, len(scores)):
            op = ops[i - 1]
            s = scores[i]
            if op == "x":
                total *= s
            else:
                total += s
        return total

    def _seat_info(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        pub = ctx.get("public") or {}
        order: List[str] = pub.get("turn_order") or []
        try:
            idx = order.index(self.name)
        except ValueError:
            idx = 0
        n = len(order) if order else 1
        return {"idx": idx, "n": n, "is_first": idx == 0, "is_last": idx == n - 1}

    def _lead_margin(self, ctx: Dict[str, Any]) -> float:
        pub = ctx.get("public") or {}
        entries = pub.get("players_public") or []
        my_total = 0
        others: List[int] = []
        for e in entries:
            nm = e.get("name")
            sc = e.get("scores") or []
            ops = e.get("ops") or []
            tot = self._eval_expr(sc, ops)
            if nm == self.name:
                my_total = tot
            else:
                others.append(tot)
        max_others = max(others) if others else 0
        return float(my_total - max_others)

    # ----------------- forbidden suit -----------------
    def choose_forbidden_suit(self, first_revealed: Card, ctx: Dict[str, Any]) -> str:
        rem: Dict[str, int] = ctx.get("deck_remaining_by_suit") or {}
        r_idx = int(ctx.get("round_index", 1))
        rounds_left = max(0, 5 - r_idx)
        seat = self._seat_info(ctx)
        lead = self._lead_margin(ctx)

        # Base weights (stats / match / anti-info / random)
        w_stats, w_match, w_anti, w_rand = 0.55, 0.20, 0.15, 0.10

        # Gates: when trailing late or last seat with few rounds left, lean harder into stats
        if lead < -5:
            w_stats += 0.10
            w_rand -= 0.05
        if seat["is_last"] and rounds_left <= 2:
            w_stats += 0.10
            w_match -= 0.05

        # Normalize
        tot = max(1e-9, w_stats + w_match + w_anti + w_rand)
        w_stats, w_match, w_anti, w_rand = (w_stats / tot, w_match / tot, w_anti / tot, w_rand / tot)

        u = random.random()
        if u < w_stats and rem and all(s in rem for s in SUITS):
            # Stats: forbid suit with FEWEST remaining (minimize p_bust)
            m = min(rem[s] for s in SUITS)
            safest = [s for s in SUITS if rem[s] == m]
            choice = random.choice(safest)
        elif u < w_stats + w_match:
            choice = first_revealed.suit
        elif u < w_stats + w_match + w_anti:
            alts = [s for s in SUITS if s != first_revealed.suit] or list(SUITS)
            choice = random.choice(alts)
        else:
            choice = random.choice(SUITS)

        self._forbidden = choice
        return choice

    # ----------------- continue / stop -----------------
    def choose_continue_or_stop(self, current_points: int, ctx: Dict[str, Any]) -> str:
        last_op = self.ops_between[-1] if self.ops_between else "+"
        rem_by: Dict[str, int] = ctx.get("deck_remaining_by_suit") or {}
        remaining = sum(rem_by.values()) if rem_by else 0
        forb = self._forbidden

        # Fallback if missing deck stats
        if remaining <= 0 or not rem_by or forb not in rem_by:
            # Conservative fallback consistent with meta (heavy @3)
            if current_points < 3:
                return "continue"
            stop_prob = 0.55 if last_op == "+" else 0.62
            return "stop" if random.random() < stop_prob else "continue"

        forbidden_left = rem_by.get(forb, 0)
        p_bust = forbidden_left / remaining if remaining > 0 else 1.0

        r_idx = int(ctx.get("round_index", 1))
        seat = self._seat_info(ctx)
        lead = self._lead_margin(ctx)

        # Minimum targets: never bank at 1–2; try not to waste × below 3 if safe
        if last_op == "+":
            if current_points < 3:
                return "continue"
        else:  # 'x'
            if current_points < 3 and p_bust <= 0.33:
                return "continue"

        # Base thresholds (stop if p_bust >= threshold); higher threshold = riskier (continue more)
        if last_op == "+":
            base = 0.95
        else:
            base = 0.78

        # Context adjustments
        # Lead/trail (per ~10 pts): trailing → raise threshold (risk more), leading → lower (risk less)
        base += 0.06 * max(0.0, -lead / 10.0)   # trailing
        base -= 0.06 * max(0.0, +lead / 10.0)   # leading

        # Round pressure: later rounds push slightly riskier if trailing, safer if leading
        if r_idx >= 4:
            base += 0.04 * (1.0 if lead < 0 else -0.04)

        # Last seat mild aggression when trailing and few rounds left
        if seat["is_last"] and r_idx >= 4 and lead < 0:
            base += 0.05

        # Final-round braking if holding a healthy lead
        if r_idx == 5 and lead >= 4:
            base -= 0.20  # more conservative at R5 with lead

        # Clamp base to reasonable band
        base = max(0.55, min(base, 1.20))

        # Translate to threshold by points (@3 ≈ main decision)
        threshold = base / (current_points + 1.0)

        # Safe-shoe nudges
        if last_op == "+" and p_bust <= 0.18:
            return "continue"
        if last_op == "x" and current_points == 3 and p_bust <= 0.28:
            return "continue"

        # Bank-5 bias in marginal zone (prefers banking at 5 when close)
        jitter = random.uniform(-0.015, 0.015)
        margin = p_bust - (threshold + jitter)
        if current_points == 5 and -0.02 <= margin <= 0.02:
            bias = 0.60 if last_op == "+" else 0.50
            return "stop" if random.random() < bias else "continue"

        return "stop" if p_bust >= (threshold + jitter) else "continue"

    # ----------------- operator between rounds -----------------
    def choose_operator_between_rounds(
        self,
        my_scores: List[int],
        all_scores: Dict[str, List[int]],
        previous_picks: List[Dict[str, str]],
        ctx: Dict[str, Any],
    ) -> str:
        last_score = my_scores[-1] if my_scores else 0
        if last_score == 0:
            return "+"

        r_idx = int(ctx.get("round_index", 1))
        seat = self._seat_info(ctx)
        lead = self._lead_margin(ctx)

        prev_op = self.ops_between[-1] if self.ops_between else None

        # Base x appetite by last round strength (targets ~50% × over time)
        if last_score >= 5:
            p = 0.62
        elif last_score >= 3:
            p = 0.52
        else:
            p = 0.36

        # Repeat × logic: keep heat if last was good
        if prev_op == "x":
            p = max(p, 0.55 if last_score >= 4 else 0.40)

        # Context scaling (logit-ish simple deltas)
        # Trailing → more ×; leading → less ×
        p += 0.05 * max(0.0, -lead / 10.0)
        p -= 0.06 * max(0.0, +lead / 10.0)

        # Late-round discipline when leading; aggression when trailing & last seat
        if r_idx == 5 and lead >= 4:
            p -= 0.12
        if seat["is_last"] and r_idx >= 4 and lead < 0:
            p += 0.06

        # Clamp
        p = max(0.10, min(p, 0.85))

        return "x" if random.random() < p else "+"
