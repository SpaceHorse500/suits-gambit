# smart_player.py
import random
from typing import List, Dict, Any, Optional
from cards import SUITS, Card
from players import BasePlayer

class SmartPlayer(BasePlayer):
    """
    Streaky, simple-probability bot:

    - Forbidden suit: random.
    - Continue/Stop:
        * Must continue at ≤ 2 points.
        * From 3+ points, stops with high and increasing probability:
            stop_prob(3) = 0.80, then +0.30 per extra point (capped at 0.99).
              e.g., 3→80%, 4→110%→cap to 99%, 5+→99%.
    - Operator choice between rounds:
        * If last round's score == 0 -> always '+' (never multiply from zero).
        * Otherwise pick randomly, BUT if this bot's previous operator was 'x',
          the chance to pick 'x' again is only 10% (to avoid long x-chains).
          If previous op wasn't 'x' (or there is none), choose 50/50.
    """
    def __init__(self, name: str):
        super().__init__(name)
        self._forbidden: Optional[str] = None

    # -------- Forbidden suit --------
    def choose_forbidden_suit(self, first_revealed: Card, ctx: Dict[str, Any]) -> str:
        # Pure random as requested
        choice = random.choice(SUITS)
        self._forbidden = choice
        return choice

    # -------- Continue / Stop --------
    def choose_continue_or_stop(self, current_points: int, ctx: Dict[str, Any]) -> str:
        # Engine enforces the mandatory first guess, but we mirror intent here.
        if current_points <= 2:
            return "continue"

        # From 3+, stop with 80% then +30% per extra point, capped at 99%
        # 3 -> 0.80, 4 -> min(1.10, 0.99)=0.99, 5+ -> 0.99
        extra = max(0, current_points - 3)
        stop_prob = min(0.80 + 0.30 * extra, 0.99)

        return "stop" if random.random() < stop_prob else "continue"

    # -------- Operator between rounds --------
    def choose_operator_between_rounds(
        self,
        my_scores: List[int],
        all_scores: Dict[str, List[int]],
        previous_picks: List[Dict[str, str]],
        ctx: Dict[str, Any],
    ) -> str:
        # Never multiply from zero (safe-guard)
        last_round_score = my_scores[-1] if my_scores else 0
        if last_round_score == 0:
            return "+"

        # Look at this bot's own previous operator (if any)
        prev_op = self.ops_between[-1] if self.ops_between else None

        if prev_op == "x":
            # After picking 'x', heavily bias away from another 'x'
            return "x" if random.random() < 0.10 else "+"
        else:
            # Otherwise, pick randomly
            return random.choice(["+", "x"])
