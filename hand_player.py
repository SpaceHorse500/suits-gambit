# smart_player.py
import random
from typing import List, Dict, Any, Optional
from cards import SUITS, Card
from players import BasePlayer


class HandPlayer(BasePlayer):
    """
    HandPlayer v4 — safer forbidden suit, smarter × segments, healthier × frequency.

    Forbidden suit (weighted modes):
      - 50%: Stats mode → choose among suit(s) with FEWEST remaining (i.e., MOST revealed).
      - 25%: Match info card suit.
      - 25%: Pure random.

    Continue/Stop (risk- & operator-aware using actual deck odds):
      - Compute p_bust = remaining(forbidden) / remaining(deck).
      - Under '+':
          * Minimum target: push to at least 3 points.
          * EV-ish threshold: continue if p_bust < 1/(points+1).
          * Extra push when risk is very low: if p_bust <= 0.18, keep going even if threshold says stop.
      - Under 'x':
          * Minimum target is soft: if points < 3 and p_bust <= 0.35 → continue (avoid wasting × on a 2).
          * General threshold: continue if p_bust < 0.80/(points+1).
      - ±0.02 jitter to avoid being perfectly predictable.
      - Reasonable fallback if deck stats missing.

    Operator between rounds (raise × frequency sensibly):
      - If last round == 0 → always '+' (never multiply zero).
      - If previous op was 'x' → 25% chance to 'x' again if last_round_score ≥ 5, else 10%.
      - Otherwise (starting a new ×-block):
            last_round_score ≥ 5 → 60% 'x'
            last_round_score 3–4 → 45% 'x'
            last_round_score ≤ 2 → 30% 'x'
    """

    def __init__(self, name: str):
        super().__init__(name)
        self._forbidden: Optional[str] = None

    # -------- Forbidden suit (weighted modes: stats 50%, match 25%, random 25%) --------
    def choose_forbidden_suit(self, first_revealed: Card, ctx: Dict[str, Any]) -> str:
        r = random.random()
        rem: Dict[str, int] = ctx.get("deck_remaining_by_suit", {}) or {}

        if r < 0.50 and rem and all(s in rem for s in SUITS):
            # Stats mode: pick suit(s) with FEWEST remaining (safest to forbid)
            min_left = min(rem[s] for s in SUITS)
            safest = [s for s in SUITS if rem[s] == min_left]
            choice = random.choice(safest)
        elif r < 0.75:
            # Match info card suit
            choice = first_revealed.suit
        else:
            # Random
            choice = random.choice(SUITS)

        self._forbidden = choice
        return choice

    # -------- Continue / Stop (deck-odds + operator aware) --------
    def choose_continue_or_stop(self, current_points: int, ctx: Dict[str, Any]) -> str:
        # Last operator (between previous round and this one); default '+' on Round 1
        last_op = self.ops_between[-1] if self.ops_between else "+"

        rem_by = ctx.get("deck_remaining_by_suit") or {}
        remaining = sum(rem_by.values()) if rem_by else 0
        forb = self._forbidden
        n_correct = max(0, current_points - 1)

        # Fallback if we somehow lack deck stats
        if remaining <= 0 or not rem_by or forb not in rem_by:
            if last_op == "x":
                base, inc = 0.68, 0.06
                if current_points < 3:
                    return "continue"
            else:
                base, inc = 0.30, 0.14
                if current_points < 3:
                    return "continue"
            stop_prob = min(0.99, base + inc * max(0, n_correct - 1))
            return "stop" if random.random() < stop_prob else "continue"

        forbidden_left = rem_by.get(forb, 0)
        p_bust = forbidden_left / remaining if remaining > 0 else 1.0
        jitter = random.uniform(-0.02, 0.02)

        if last_op == "x":
            # Don’t waste × on a tiny segment: try to reach 3 when risk allows
            if current_points < 3 and p_bust <= 0.35:
                return "continue"
            # General × threshold (a bit braver than v3)
            p_thresh = 0.80 / (current_points + 1)
            return "stop" if p_bust >= (p_thresh + jitter) else "continue"

        else:
            # Addition: push to at least 3
            if current_points < 3:
                return "continue"
            # EV-ish threshold
            p_thresh = 1.0 / (current_points + 1)
            # Extra push when the shoe is very safe
            if p_bust <= 0.18:
                return "continue"
            return "stop" if p_bust >= (p_thresh + jitter) else "continue"

    # -------- Operator between rounds (score-aware × appetite) --------
    def choose_operator_between_rounds(
        self,
        my_scores: List[int],
        all_scores: Dict[str, List[int]],
        previous_picks: List[Dict[str, str]],
        ctx: Dict[str, Any],
    ) -> str:
        last_round_score = my_scores[-1] if my_scores else 0
        if last_round_score == 0:
            return "+"

        prev_op = self.ops_between[-1] if self.ops_between else None
        if prev_op == "x":
            # Allow some ×→× if last was strong
            repeat_prob = 0.25 if last_round_score >= 5 else 0.10
            return "x" if random.random() < repeat_prob else "+"

        # Starting a new ×-block: raise × usage into ~25–30%+ range overall
        if last_round_score >= 5:
            x_prob = 0.60
        elif last_round_score >= 3:
            x_prob = 0.45
        else:
            x_prob = 0.30

        return "x" if random.random() < x_prob else "+"
